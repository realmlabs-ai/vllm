# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission-level integration test for the SAE scorer's client spec.

Confirms ``SaeScorerConsumer.validate_client_spec`` actually plugs into the
shared ``resolve_capture_prefix_flags`` pipeline (``vllm/v1/capture/
admission.py``) the way ``FilesystemConsumer``'s already does — this is the
mechanism that makes the consumer's tap visible to the scheduler's automatic
prefix-caching (APC) decision (see ``consumer.py``'s ``reads_client_spec``
docstring, and ``DELIVERY_NOTES.md``, for the "Bug B" this fixes).

``test_admission.py`` covers the shared resolution logic with hand-rolled
consumer doubles; this module exercises the same entry point with the real
``SaeScorerConsumer`` so a regression in either side (the consumer's spec, or
admission's floor computation) is caught.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm.sampling_params import SamplingParams
from vllm.v1.capture.admission import resolve_capture_prefix_flags
from vllm.v1.capture.consumers.sae_scorer import consumer as sae_consumer
from vllm.v1.capture.consumers.sae_scorer.consumer import SaeScorerConsumer
from vllm.v1.capture.types import CaptureContext, VllmInternalRequestId

HIDDEN = 8
SAE_SIZE = 16


def _make_consumer(monkeypatch, *, params=None):
    """Same helper shape as ``test_sae_scorer.py`` (kept local — this module
    should be runnable standalone without importing test internals from
    another test file)."""
    from .test_sae_scorer import _make_tiny_sae

    sae = _make_tiny_sae()
    monkeypatch.setattr(sae_consumer, "load_sae", lambda **kw: sae)
    vllm_config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"))
    full_params = {"device": "cpu"}
    full_params.update(params or {})
    return SaeScorerConsumer(vllm_config, full_params)


def _ctx(num_prompt_tokens: int = 8) -> CaptureContext:
    return CaptureContext(
        vllm_internal_request_id=VllmInternalRequestId("req-1"),
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=0,
        num_hidden_layers=32,
        hidden_size=HIDDEN,
        element_size_bytes=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


class TestSaeScorerAdmission:
    def test_default_all_positions_stamps_full_prompt_floor(self, monkeypatch):
        """Default ``positions="all"`` taps the whole prompt, so the
        request-wide floor is 0 — the whole prompt must re-forward
        (and thus capture), same as filesystem's ``all_prompt`` case."""
        consumer = _make_consumer(monkeypatch)
        sp = SamplingParams(capture={"sae_scorer": {}})
        resolve_capture_prefix_flags({"sae_scorer": consumer}, sp, _ctx(8))

        assert sp.capture_touches_prompt is True
        assert sp.capture_min_prompt_position == 0
        assert sp.capture_store_hook_layers == [("post_mlp", 20)]
        assert sp.capture_store_positions == list(range(8))

    def test_last_prompt_override_floors_at_final_position(self, monkeypatch):
        consumer = _make_consumer(monkeypatch)
        sp = SamplingParams(capture={"sae_scorer": {"positions": "last_prompt"}})
        resolve_capture_prefix_flags({"sae_scorer": consumer}, sp, _ctx(8))

        assert sp.capture_touches_prompt is True
        assert sp.capture_min_prompt_position == 7
        assert sp.capture_store_positions == [7]

    def test_generated_only_keeps_full_prefix_caching(self, monkeypatch):
        """The whole point of this fix: once the consumer's tap is
        classifiable, a generated-only spec is correctly recognized as never
        touching the prompt, and prefix caching is left fully enabled for it
        — this consumer never forces a re-forward it doesn't need."""
        consumer = _make_consumer(
            monkeypatch, params={"positions": "all_generated"}
        )
        sp = SamplingParams(capture={"sae_scorer": {}})
        resolve_capture_prefix_flags({"sae_scorer": consumer}, sp, _ctx(8))

        assert sp.capture_touches_prompt is False
        assert sp.capture_min_prompt_position is None
        assert sp.capture_store_hook_layers is None
        assert sp.capture_store_positions is None

    def test_not_opting_in_leaves_flags_unset(self, monkeypatch):
        """Without a ``capture={"sae_scorer": ...}`` entry, admission never
        sees this consumer at all — the exact pre-fix behavior for a request
        that only relies on the (still-supported) global spec. Confirms the
        fix is additive: it does not change behavior for requests that don't
        opt in."""
        consumer = _make_consumer(monkeypatch)
        sp = SamplingParams(capture=None)
        resolve_capture_prefix_flags({"sae_scorer": consumer}, sp, _ctx(8))

        assert sp.capture_touches_prompt is None
        assert sp.capture_min_prompt_position is None
