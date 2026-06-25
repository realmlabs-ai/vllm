# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Early finalize: a request whose every consumer captures only prompt
positions is data-complete at end of prefill, so the manager flags it for
immediate finalize (``take_prompt_complete_requests``) instead of waiting for
the post-finish step. Generated-token specs are never early-flagged.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from vllm.v1.capture.manager import CaptureManager
from vllm.v1.capture.plan import CaptureBatchView
from vllm.v1.capture.types import CaptureSpec

NUM_LAYERS = 4
HIDDEN_SIZE = 8
MODEL_DTYPE = torch.float32


def _sink() -> MagicMock:
    s = MagicMock()
    s.location = "worker"
    s.wants_device_tensors = False
    s.submit_chunk = MagicMock()
    s.submit_chunk_batch = None
    s.submit_finalize = MagicMock()
    s.get_result = MagicMock(return_value=None)
    s.wait_for_result = MagicMock(return_value=None)
    s.shutdown = MagicMock()
    return s


def _mgr(specs):
    sinks = tuple(_sink() for _ in specs)
    return CaptureManager(
        consumers=sinks,
        consumer_specs=tuple(specs),
        num_hidden_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        model_dtype=MODEL_DTYPE,
    )


def _view(num_prompt, num_computed, num_scheduled):
    return CaptureBatchView(
        req_ids=["r1"],
        num_prompt_tokens=[num_prompt],
        num_computed_tokens=[num_computed],
        num_scheduled_tokens=[num_scheduled],
        token_offsets=[0],
    )


def _spec(positions):
    return CaptureSpec(hooks={"post_mlp": [0]}, positions=positions)


class TestEarlyFinalize:
    def test_prompt_bounded_flagged_after_prefill(self):
        mgr = _mgr([_spec("all_prompt")])
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        # Nothing ready before any step.
        assert mgr.take_prompt_complete_requests() == []
        # Full prefill in one step => prompt complete.
        mgr.build_step_plan(_view(10, 0, 10))
        assert mgr.take_prompt_complete_requests() == ["r1"]
        # Cleared after taking.
        assert mgr.take_prompt_complete_requests() == []

    def test_last_prompt_flagged(self):
        mgr = _mgr([_spec("last_prompt")])
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        mgr.build_step_plan(_view(10, 0, 10))
        assert mgr.take_prompt_complete_requests() == ["r1"]

    def test_chunked_prefill_flags_only_when_complete(self):
        mgr = _mgr([_spec("all_prompt")])
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        # First chunk: prompt not yet fully forwarded.
        mgr.build_step_plan(_view(10, 0, 6))
        assert mgr.take_prompt_complete_requests() == []
        # Second chunk completes the prompt.
        mgr.build_step_plan(_view(10, 6, 4))
        assert mgr.take_prompt_complete_requests() == ["r1"]

    def test_generated_spec_never_flagged(self):
        mgr = _mgr([_spec("all")])
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        mgr.build_step_plan(_view(10, 0, 10))  # prompt forwarded
        assert mgr.take_prompt_complete_requests() == []
        mgr.build_step_plan(_view(10, 10, 1))  # a decode step
        assert mgr.take_prompt_complete_requests() == []

    def test_mixed_consumers_not_bounded(self):
        # One prompt consumer + one generated consumer => request not bounded.
        mgr = _mgr([_spec("all_prompt"), _spec("all_generated")])
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        mgr.build_step_plan(_view(10, 0, 10))
        assert mgr.take_prompt_complete_requests() == []

    def test_explicit_prompt_positions_bounded(self):
        mgr = _mgr([_spec([2, 5, 9])])
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        mgr.build_step_plan(_view(10, 0, 10))
        assert mgr.take_prompt_complete_requests() == ["r1"]

    def test_explicit_generated_position_not_bounded(self):
        # Position 12 is beyond the 10-token prompt => not prompt-bounded.
        mgr = _mgr([_spec([2, 12])])
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        mgr.build_step_plan(_view(10, 0, 10))
        assert mgr.take_prompt_complete_requests() == []

    def test_finalized_request_filtered_out(self):
        mgr = _mgr([_spec("all_prompt")])
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        mgr.build_step_plan(_view(10, 0, 10))
        # Finalize (pops the request) before draining the flag.
        mgr.finalize_request("r1")
        assert mgr.take_prompt_complete_requests() == []
