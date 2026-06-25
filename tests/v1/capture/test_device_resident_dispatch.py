# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the device-resident dispatch path in ``CaptureManager``.

A consumer that sets ``wants_device_tensors = True`` should receive the
on-device ``scratch_gpu`` rows *synchronously* from
:meth:`CaptureManager.dispatch_step_captures` — with no host (D2H) copy and
no trip through the async dispatch thread. When *only* device-resident
consumers want a step's rows, the manager skips the pinned-host /
``cuda.Event`` / packet path entirely (the Option-C efficiency win). Host
consumers keep their existing async, host-resident delivery.

These tests run on CPU (``scratch_gpu`` holds CPU tensors), so they exercise
the routing/partition logic without a GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from vllm.v1.capture.manager import CaptureManager
from vllm.v1.capture.plan import CaptureBatchView
from vllm.v1.capture.types import CaptureResult, CaptureSpec

NUM_LAYERS = 4
HIDDEN_SIZE = 8
MODEL_DTYPE = torch.float32


class _RecordingSink:
    """A minimal ``CaptureSink`` that records the chunks/finalizes it gets.

    Unlike a bare ``MagicMock`` it exposes ``wants_device_tensors`` as a real
    bool, so the manager's ``getattr(sink, "wants_device_tensors", False) is
    True`` check classifies it deterministically.
    """

    # The manager probes ``submit_chunk_batch`` via ``getattr``; ``None`` forces
    # the per-chunk path so the recorded list is one-entry-per-chunk.
    submit_chunk_batch = None

    def __init__(self, *, wants_device_tensors: bool) -> None:
        self.location = "worker"
        self.wants_device_tensors = wants_device_tensors
        self.chunks: list = []
        self.finalizes: list = []
        self._results: dict = {}

    def submit_chunk(self, chunk) -> None:
        self.chunks.append(chunk)

    def submit_finalize(self, finalize) -> None:
        self.finalizes.append(finalize)
        self._results[finalize.key] = CaptureResult(key=finalize.key, status="ok")

    def get_result(self, key):
        return self._results.get(key)

    def wait_for_result(self, key, timeout):
        return self._results.get(key)

    def shutdown(self, timeout: float = 30.0) -> None:
        pass


def _make_host_sink() -> MagicMock:
    """A host (non-device) mock sink, mirroring tests/v1/capture/test_manager.py."""
    sink = MagicMock()
    sink.location = "worker"
    sink.wants_device_tensors = False
    sink.submit_chunk = MagicMock()
    sink.submit_chunk_batch = None
    sink.submit_finalize = MagicMock()
    sink.get_result = MagicMock(return_value=None)
    sink.wait_for_result = MagicMock(return_value=None)
    sink.shutdown = MagicMock()
    return sink


def _spec() -> CaptureSpec:
    return CaptureSpec(hooks={"post_mlp": [0, 1]}, positions="last_prompt")


def _batch_view() -> CaptureBatchView:
    return CaptureBatchView(
        req_ids=["r1"],
        num_prompt_tokens=[10],
        num_computed_tokens=[0],
        num_scheduled_tokens=[10],
        token_offsets=[0],
    )


def _populate_scratch(plan) -> None:
    """Mimic ``on_hook``: fill ``scratch_gpu`` with deterministic rows."""
    for key, idx in plan.gather_indices.items():
        n_rows = idx.shape[0]
        scratch = torch.zeros((n_rows, HIDDEN_SIZE), dtype=MODEL_DTYPE)
        for r in range(n_rows):
            scratch[r] = torch.arange(HIDDEN_SIZE, dtype=MODEL_DTYPE) + r * 100
        plan.scratch_gpu[key] = scratch


def _make_manager(sinks, specs) -> CaptureManager:
    return CaptureManager(
        consumers=tuple(sinks),
        consumer_specs=tuple(specs),
        num_hidden_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        model_dtype=MODEL_DTYPE,
    )


class TestDeviceResidentDispatch:
    def test_device_only_delivers_synchronously_and_skips_d2h(self, monkeypatch):
        dev = _RecordingSink(wants_device_tensors=True)
        mgr = _make_manager([dev], [_spec()])

        # Spy on the async-dispatch enqueue: a GPU-only step must skip the
        # entire host path (no packet, hence no D2H / pinned-host / cuda.Event).
        enqueue_calls: list = []
        orig = mgr._enqueue_packet
        monkeypatch.setattr(
            mgr,
            "_enqueue_packet",
            lambda pkt: (enqueue_calls.append(pkt), orig(pkt))[1],
        )

        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        plan = mgr.build_step_plan(_batch_view())
        _populate_scratch(plan)

        mgr.dispatch_step_captures(plan)

        # Delivered synchronously — no _drain_dispatch_queue() needed.
        assert len(dev.chunks) == 2
        # No packet was queued for the async dispatch thread (host path skipped).
        assert enqueue_calls == []
        assert mgr._dispatch_queue.qsize() == 0

        # Values are the scratch rows (last_prompt => row 0 == arange(HIDDEN)).
        by_layer = {c.key[1]: c for c in dev.chunks}
        assert set(by_layer) == {0, 1}
        for chunk in dev.chunks:
            assert chunk.key[2] == "post_mlp"
            assert chunk.tensor.shape == (1, HIDDEN_SIZE)
            assert torch.equal(
                chunk.tensor[0], torch.arange(HIDDEN_SIZE, dtype=MODEL_DTYPE)
            )

        results = mgr.finalize_request("r1")
        assert 0 in results
        assert len(dev.finalizes) == 2

    def test_mixed_host_and_device(self, monkeypatch):
        host = _make_host_sink()
        dev = _RecordingSink(wants_device_tensors=True)
        mgr = _make_manager([host, dev], [_spec(), _spec()])

        enqueue_calls: list = []
        orig = mgr._enqueue_packet
        monkeypatch.setattr(
            mgr,
            "_enqueue_packet",
            lambda pkt: (enqueue_calls.append(pkt), orig(pkt))[1],
        )

        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        plan = mgr.build_step_plan(_batch_view())
        _populate_scratch(plan)

        mgr.dispatch_step_captures(plan)

        # Device sink got its rows synchronously.
        assert len(dev.chunks) == 2
        # Host sink is async — nothing until the dispatch thread runs.
        assert host.submit_chunk.call_count == 0
        # Host path DID enqueue a packet (host consumer present).
        assert len(enqueue_calls) == 1

        mgr._drain_dispatch_queue()
        assert host.submit_chunk.call_count == 2

        # Host chunks must not be device-routed duplicates: each sink sees its
        # own two chunks, no more.
        assert len(dev.chunks) == 2

    def test_no_device_consumers_uses_pure_host_path(self, monkeypatch):
        host = _make_host_sink()
        mgr = _make_manager([host], [_spec()])

        enqueue_calls: list = []
        orig = mgr._enqueue_packet
        monkeypatch.setattr(
            mgr,
            "_enqueue_packet",
            lambda pkt: (enqueue_calls.append(pkt), orig(pkt))[1],
        )

        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        plan = mgr.build_step_plan(_batch_view())
        _populate_scratch(plan)

        mgr.dispatch_step_captures(plan)
        mgr._drain_dispatch_queue()

        assert host.submit_chunk.call_count == 2
        assert len(enqueue_calls) == 1  # unchanged host behaviour
