# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Store mode (solution 2) for ``SaeScorerConsumer``.

With ``out_dir`` set, ``on_capture`` writes the full sparse scores to a
deterministic per-request path and returns only a small handle. The on-disk
file is the source of truth; the handle is a convenience and must stay
msgspec-serializable.
"""

from __future__ import annotations

import json
import os

import msgspec
import pytest
import torch

from vllm.v1.capture.consumers.sae_scorer import consumer as sae_consumer
from vllm.v1.capture.consumers.sae_scorer import score_path
from vllm.v1.capture.consumers.sae_scorer.consumer import (
    DEFAULT_SAE_SRC,
    SaeScorerConsumer,
)
from vllm.v1.capture.types import CaptureResult, VllmInternalRequestId

HIDDEN = 8
SAE_SIZE = 16

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_SAE_SRC),
    reason=f"ml_service SAE module not present at {DEFAULT_SAE_SRC}",
)


def _make_tiny_sae():
    mod = sae_consumer._import_sae_module(DEFAULT_SAE_SRC)
    cfg = mod.SAESignalConfig(
        signal_name="tiny",
        target_layer_index=20,
        hidden_size=HIDDEN,
        sae_size=SAE_SIZE,
        activation="jumprelu_uniform",
        threshold=0.0,
        dtype=torch.float32,
    )
    sae = mod.SAESignal(config=cfg)
    torch.manual_seed(0)
    with torch.no_grad():
        sae.encoder.weight.normal_()
        sae.encoder.bias.normal_()
        sae.threshold.fill_(0.0)
    sae.eval()
    return sae


def _make_consumer(monkeypatch, out_dir):
    sae = _make_tiny_sae()
    monkeypatch.setattr(sae_consumer, "load_sae", lambda **kw: sae)
    from types import SimpleNamespace

    vllm_config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"))
    consumer = SaeScorerConsumer(
        vllm_config, {"device": "cpu", "out_dir": str(out_dir)}
    )
    return consumer, sae


def _key(rid="req-1"):
    return (VllmInternalRequestId(rid), 20, "post_mlp")


class TestStoreMode:
    def test_writes_file_at_deterministic_path_and_returns_handle(
        self, monkeypatch, tmp_path
    ):
        consumer, sae = _make_consumer(monkeypatch, tmp_path)
        x = torch.randn(3, HIDDEN)

        handle = consumer.on_capture(_key("req-1"), x, {})

        # Handle is a small pointer, NOT the full scores.
        assert handle["stored"] is True
        assert handle["request_id"] == "req-1"
        assert handle["layer"] == 20
        assert handle["num_rows"] == 3
        assert "indices" not in handle and "values" not in handle

        # The deterministic path a client would compute itself.
        expected = score_path(tmp_path, "req-1", 20, "post_mlp")
        assert handle["path"] == str(expected)
        assert expected.exists()

        # File content reconstructs the SAE encode exactly.
        stored = json.loads(expected.read_text())
        assert stored["sae_size"] == SAE_SIZE
        assert stored["num_rows"] == 3
        ref = sae.encode(x)
        for r in range(3):
            ref_idx = ref[r].nonzero(as_tuple=True)[0].tolist()
            assert stored["indices"][r] == ref_idx

    def test_handle_is_msgspec_serializable(self, monkeypatch, tmp_path):
        consumer, _ = _make_consumer(monkeypatch, tmp_path)
        handle = consumer.on_capture(_key(), torch.randn(2, HIDDEN), {})
        result = CaptureResult(key=_key(), status="ok", payload=handle)
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(result), type=CaptureResult
        )
        assert decoded.payload["stored"] is True
        assert decoded.payload["path"] == handle["path"]

    def test_no_partial_file_on_reader_view(self, monkeypatch, tmp_path):
        # After on_capture returns, only the final file exists (no .tmp left).
        consumer, _ = _make_consumer(monkeypatch, tmp_path)
        consumer.on_capture(_key("req-x"), torch.randn(1, HIDDEN), {})
        files = list((tmp_path / "req-x").iterdir())
        assert [f.name for f in files] == ["L20_post_mlp.json"]

    def test_empty_capture_still_writes_wellformed_file(self, monkeypatch, tmp_path):
        consumer, _ = _make_consumer(monkeypatch, tmp_path)
        handle = consumer.on_capture(_key("req-0"), torch.empty((0, HIDDEN)), {})
        assert handle["num_rows"] == 0
        stored = json.loads(score_path(tmp_path, "req-0", 20, "post_mlp").read_text())
        assert stored["num_rows"] == 0
        assert stored["indices"] == [] and stored["values"] == []

    def test_request_id_is_slugged_for_filesystem_safety(self, monkeypatch, tmp_path):
        consumer, _ = _make_consumer(monkeypatch, tmp_path)
        # A req id with a slash must not escape out_dir.
        handle = consumer.on_capture(_key("0/../etc"), torch.randn(1, HIDDEN), {})
        path = score_path(tmp_path, "0/../etc", 20, "post_mlp")
        assert handle["path"] == str(path)
        assert path.exists()
        assert tmp_path in path.parents  # stayed inside out_dir
