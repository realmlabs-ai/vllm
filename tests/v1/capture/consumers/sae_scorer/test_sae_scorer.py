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
from vllm.v1.capture.types import VllmInternalRequestId

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
