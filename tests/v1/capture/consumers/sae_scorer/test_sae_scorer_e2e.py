# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end: run llama-3.1-8B + the L20 SAE scorer and read scores back.

Requires a CUDA GPU, the llama-3.1-8B-instruct weights, and the L20 SAE — so it
``skip``s when any are absent. It asserts the *structure* of the returned SAE
scores (layer/sae_size, per-position sparse features in range), not their
semantics (the SAE was trained on the base model; instruct activations drift).

Delivery note: capture finalize is triggered on the step *after* a request
finishes, which the offline engine never runs for the last/only request — so
``RequestOutput.capture_results`` is empty for a single short generation (a
known limitation of the current branch; the reverted ``capture-wait`` feature
addressed it). The captured rows ARE delivered to the device-resident consumer
during the forward; we retrieve the scores via the supported synchronous
``CaptureManager.finalize_request`` (the same API the runner unit tests use),
which is the reliable in-process path today. Runs in-process
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``) so the manager is reachable.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile

# Must be set before importing vllm: run the engine in-process (so the capture
# manager is reachable) and use the native sampler (no flashinfer JIT/ninja).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import pytest
import torch

from vllm.v1.capture.consumers.sae_scorer.consumer import (
    DEFAULT_SAE_DIR,
    DEFAULT_SAE_SRC,
)

MODEL = "/home/mayank/research/models/llama-3.1-8B-instruct"
SIGNAL_NAME = "l20_res_8x"
LAYER = 20
SAE_SIZE = 32768
OUT_DIR = tempfile.mkdtemp(prefix="vllm-sae-e2e-")

_missing = [
    p for p in (MODEL, DEFAULT_SAE_SRC, DEFAULT_SAE_DIR) if not os.path.exists(p)
]
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(bool(_missing), reason=f"missing artifacts: {_missing}"),
]


@pytest.fixture(scope="module")
def llm():
    from vllm import LLM

    engine = LLM(
        model=MODEL,
        enforce_eager=True,  # avoid cudagraph/device-path interaction for MVP
        gpu_memory_utilization=0.5,  # leave room; SAE adds ~0.5 GB
        max_model_len=2048,
        capture_consumers=[
            {
                "name": "sae_scorer",
                "params": {
                    "layer": LAYER,
                    "signal_name": SIGNAL_NAME,
                    "positions": "all_prompt",
                    "out_dir": OUT_DIR,  # solution 2: write scores to disk
                },
            }
        ],
    )
    yield engine
    del engine


def _validate_store_file(path: pathlib.Path):
    assert path.exists(), f"score file not written: {path}"
    stored = json.loads(path.read_text())
    assert stored["sae_size"] == SAE_SIZE
    assert stored["num_rows"] >= 1
    for row_idx, row_vals in zip(stored["indices"], stored["values"]):
        assert len(row_idx) == len(row_vals)
        assert all(0 <= i < SAE_SIZE for i in row_idx)
        assert all(v > 0 for v in row_vals)


def test_sae_scores_written_automatically(llm):
    """Solution 2 + early-finalize: scores hit the store with NO manual finalize.

    ``positions="all_prompt"`` is prompt-bounded, so the manager early-finalizes
    the request after prefill (while it is still generating) — the consumer
    writes the store file during generation, with no dependence on the
    post-finish finalize step. The store file is the reliable contract; the
    inline ``out.capture_results`` handle is a best-effort bonus (the racy
    delivery path documented in DELIVERY_NOTES.md), so it is not required here.
    """
    from vllm import SamplingParams

    outs = llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0.0, max_tokens=8),
    )
    out = outs[0]

    # Reliable: the score file was written automatically during generation.
    files = sorted(pathlib.Path(OUT_DIR).glob("*/L20_post_mlp.json"))
    assert files, f"no score file written under {OUT_DIR}"
    _validate_store_file(files[0])

    # Best-effort bonus: if the handle rode the response, it must be well-formed.
    if "sae_scorer" in out.capture_results:
        payload = out.capture_results["sae_scorer"].payload
        assert payload["stored"] is True
        assert payload["layer"] == LAYER
