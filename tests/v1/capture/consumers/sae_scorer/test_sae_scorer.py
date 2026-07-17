# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``SaeScorerConsumer`` (CPU, tiny SAE, no model).

These build a small real ``SAESignal`` from the ml_service ``sae.py`` (the same
module the consumer loads in production) with random weights, monkeypatch the
consumer's ``load_sae`` to return it, and check that ``on_capture`` returns
sparse scores that exactly reconstruct ``sae.encode``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.capture.consumers.sae_scorer import consumer as sae_consumer
from vllm.v1.capture.consumers.sae_scorer.consumer import (
    DEFAULT_SAE_SRC,
    SaeScorerConsumer,
)
from vllm.v1.capture.errors import CaptureValidationError
from vllm.v1.capture.types import (
    CaptureContext,
    VllmInternalRequestId,
    min_captured_prompt_position,
)

HIDDEN = 8
SAE_SIZE = 16

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_SAE_SRC),
    reason=f"ml_service SAE module not present at {DEFAULT_SAE_SRC}",
)


def _make_tiny_sae(threshold: float = 0.0):
    """Build a small real ``SAESignal`` (jumprelu_uniform) with random weights."""
    mod = sae_consumer._import_sae_module(DEFAULT_SAE_SRC)
    cfg = mod.SAESignalConfig(
        signal_name="tiny",
        target_layer_index=20,
        hidden_size=HIDDEN,
        sae_size=SAE_SIZE,
        activation="jumprelu_uniform",
        threshold=threshold,
        dtype=torch.float32,
    )
    sae = mod.SAESignal(config=cfg)
    torch.manual_seed(0)
    with torch.no_grad():
        sae.encoder.weight.normal_()
        sae.encoder.bias.normal_()
        sae.threshold.fill_(threshold)
    sae.eval()
    return sae


def _make_consumer(monkeypatch, *, sae=None, params=None):
    sae = sae or _make_tiny_sae()
    monkeypatch.setattr(sae_consumer, "load_sae", lambda **kw: sae)
    vllm_config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"))
    full_params = {"device": "cpu"}
    full_params.update(params or {})
    return SaeScorerConsumer(vllm_config, full_params), sae


def _key():
    return (VllmInternalRequestId("r1"), 20, "post_mlp")


def _ctx(num_prompt_tokens: int = 8, num_hidden_layers: int = 32) -> CaptureContext:
    return CaptureContext(
        vllm_internal_request_id=VllmInternalRequestId("r1"),
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=0,
        num_hidden_layers=num_hidden_layers,
        hidden_size=HIDDEN,
        element_size_bytes=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


class TestSaeScorerConsumer:
    def test_global_capture_spec_defaults(self, monkeypatch):
        consumer, _ = _make_consumer(monkeypatch)
        spec = consumer.global_capture_spec()
        assert spec.hooks == {"post_mlp": [20]}
        assert spec.positions == "all"

    def test_global_capture_spec_configurable(self, monkeypatch):
        consumer, _ = _make_consumer(
            monkeypatch,
            params={"layer": 3, "hook": "pre_attn", "positions": "last_prompt"},
        )
        spec = consumer.global_capture_spec()
        assert spec.hooks == {"pre_attn": [3]}
        assert spec.positions == "last_prompt"

    def test_wants_device_tensors(self):
        assert SaeScorerConsumer.wants_device_tensors is True

    def test_on_capture_sparse_matches_encode(self, monkeypatch):
        consumer, sae = _make_consumer(monkeypatch)
        x = torch.randn(3, HIDDEN)

        payload = consumer.on_capture(_key(), x, {})

        assert payload["layer"] == 20
        assert payload["hook"] == "post_mlp"
        assert payload["sae_size"] == SAE_SIZE
        assert payload["num_rows"] == 3
        assert len(payload["indices"]) == 3
        assert len(payload["values"]) == 3

        ref = sae.encode(x)  # (3, SAE_SIZE)
        for r in range(3):
            ref_idx = ref[r].nonzero(as_tuple=True)[0].tolist()
            assert payload["indices"][r] == ref_idx
            ref_vals = ref[r][ref[r].nonzero(as_tuple=True)[0]].tolist()
            assert payload["values"][r] == pytest.approx(ref_vals, rel=1e-5)
            # Every reported feature is genuinely active (sparsity sanity).
            assert all(v > 0 for v in payload["values"][r])

    def test_skip_first_emits_empty_row0(self, monkeypatch):
        consumer, sae = _make_consumer(monkeypatch, params={"skip_first": True})
        x = torch.randn(3, HIDDEN)
        payload = consumer.on_capture(_key(), x, {})
        assert payload["num_rows"] == 3                 # row kept for alignment
        assert payload["indices"][0] == [] and payload["values"][0] == []  # BOS skipped
        ref = sae.encode(x)
        for r in (1, 2):                                # other rows unaffected
            assert payload["indices"][r] == ref[r].nonzero(as_tuple=True)[0].tolist()

    def test_on_capture_empty(self, monkeypatch):
        consumer, _ = _make_consumer(monkeypatch)
        payload = consumer.on_capture(_key(), torch.empty((0, HIDDEN)), {})
        assert payload["num_rows"] == 0
        assert payload["indices"] == []
        assert payload["values"] == []
        assert payload["sae_size"] == SAE_SIZE

    def test_on_capture_topk_caps_features(self, monkeypatch):
        # High threshold so plain nonzero would still exceed k for some rows.
        sae = _make_tiny_sae(threshold=-10.0)  # nothing gated -> all 16 active
        consumer, _ = _make_consumer(monkeypatch, sae=sae, params={"topk": 4})
        payload = consumer.on_capture(_key(), torch.randn(5, HIDDEN), {})
        for row_idx in payload["indices"]:
            assert len(row_idx) <= 4


class TestSaeScorerClientSpec:
    """Per-request client-spec opt-in (Bug B fix — see docs/vllm/README.md).

    Without ``reads_client_spec``/``validate_client_spec``, this consumer's
    ``(layer, hook)`` tap is a pure *global* spec invisible to admission's
    prefix-cache resolution, so automatic prefix caching (APC) can silently
    skip forwarding — and therefore capturing — cached prompt positions.
    """

    def test_reads_client_spec_is_true(self):
        assert SaeScorerConsumer.reads_client_spec is True

    def test_validate_client_spec_matches_global_defaults(self, monkeypatch):
        consumer, _ = _make_consumer(monkeypatch)
        spec = consumer.validate_client_spec({}, _ctx())
        assert spec.hooks == {"post_mlp": [20]}
        assert spec.positions == "all"
        # Same floor as the (never-visible-to-admission) global spec would
        # imply, now actually reachable by resolve_capture_prefix_flags.
        assert min_captured_prompt_position(spec, num_prompt_tokens=8) == 0

    def test_validate_client_spec_accepts_none(self, monkeypatch):
        consumer, _ = _make_consumer(monkeypatch)
        spec = consumer.validate_client_spec(None, _ctx())
        assert spec.hooks == {"post_mlp": [20]}

    def test_validate_client_spec_honors_positions_override(self, monkeypatch):
        consumer, _ = _make_consumer(monkeypatch)
        spec = consumer.validate_client_spec({"positions": "last_prompt"}, _ctx())
        assert spec.positions == "last_prompt"
        assert spec.hooks == {"post_mlp": [20]}  # layer/hook stay server-configured

    def test_validate_client_spec_matches_configured_layer_hook(self, monkeypatch):
        consumer, _ = _make_consumer(
            monkeypatch, params={"layer": 3, "hook": "pre_attn"}
        )
        spec = consumer.validate_client_spec({}, _ctx())
        assert spec.hooks == {"pre_attn": [3]}

    def test_validate_client_spec_rejects_non_dict(self, monkeypatch):
        consumer, _ = _make_consumer(monkeypatch)
        with pytest.raises(CaptureValidationError):
            consumer.validate_client_spec("not-a-dict", _ctx())

    def test_validate_client_spec_rejects_out_of_range_layer(self, monkeypatch):
        consumer, _ = _make_consumer(monkeypatch, params={"layer": 40})
        with pytest.raises(CaptureValidationError):
            consumer.validate_client_spec({}, _ctx(num_hidden_layers=32))

    def test_validate_client_spec_rejects_unwired_hook(self, monkeypatch):
        consumer, _ = _make_consumer(monkeypatch, params={"hook": "mlp_in"})
        ctx = _ctx()
        # A non-empty hook_schema that doesn't include "mlp_in" (mirrors a
        # real model's schema, which lists only its wired hooks).
        ctx.hook_schema = {"post_mlp": object(), "pre_attn": object()}
        with pytest.raises(CaptureValidationError):
            consumer.validate_client_spec({}, ctx)
