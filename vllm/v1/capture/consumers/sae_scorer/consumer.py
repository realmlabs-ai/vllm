# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A device-resident capture consumer that scores hidden states with an SAE.

``SaeScorerConsumer`` requests one ``(layer, hook)`` residual via a global
capture spec, receives the captured rows **on the model device** (no D2H copy —
``wants_device_tensors = True``), and runs a Sparse Autoencoder ``encode`` on
the GPU to produce compact **sparse** feature activations ("SAE scores").

Two egress modes (see ``DELIVERY_NOTES.md`` for the full rationale):

- **Store mode (solution 2, default when ``params["out_dir"]`` is set):** the
  scores are written to disk at a *deterministic* per-request path and
  ``on_capture`` returns only a small **handle** (the path + shape). This is the
  reliable path — it does not depend on the racy ``capture_results`` delivery
  (the client reads the file by request id). It is also the only sane channel
  for large artifacts and works under streaming.
- **Inline mode (no ``out_dir``):** ``on_capture`` returns the full sparse
  scores as the payload. Convenient for in-process / early-finalize use, but
  delivery via ``RequestOutput.capture_results`` is subject to the finalize
  timing caveats documented in ``DELIVERY_NOTES.md``.

The SAE itself is the self-contained ``SAESignal`` from the user's research repo
(``ml_service``). Rather than vendoring it, we import the module from a
configurable path (``params["sae_src"]``) and load weights via its
``load_sae_local`` (``params["sae_dir"]`` + ``signal_name``). torch + pydantic
are the only deps, both already present in a vLLM environment.

Payload shape (rows are in ascending captured-position order)::

    {
        "layer": 20,
        "hook": "post_mlp",
        "sae_size": 32768,
        "num_rows": N,
        "indices": [[f0, f1, ...], ...],  # active feature ids, per row
        "values": [[v0, v1, ...], ...],  # their activations, per row
    }

JumpReLU SAEs gate most features to zero, so the per-row lists are short.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import sys
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch

from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.types import CaptureKey, CaptureSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Sensible defaults for this environment; every one is overridable via params.
DEFAULT_SAE_SRC = "/home/mayank/research/ml_service/experimental/mayank/sae.py"
DEFAULT_SAE_DIR = "/home/mayank/research/models/SAE/llama-3.1-8B-Base/signals"
DEFAULT_LAYER = 20
DEFAULT_HOOK = "post_mlp"
DEFAULT_POSITIONS: Any = "all"

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")


def _slug(value: str) -> str:
    """Filesystem-safe slug for a request id (keeps it readable).

    Maps any path-traversal or empty result to ``_`` so the slug can only ever
    be a single, in-tree directory component.
    """
    slug = _SLUG_RE.sub("_", value)
    if slug in ("", ".", ".."):
        return "_"
    return slug


def score_path(out_dir: pathlib.Path, request_id: str, layer: int, hook: str):
    """Deterministic on-disk location of a request's scores for ``(layer, hook)``.

    Clients can compute this from ``(out_dir, request_id, layer, hook)`` without
    the handle, so retrieval never depends on ``capture_results`` delivery.
    """
    return out_dir / _slug(request_id) / f"L{layer}_{hook}.json"


def _import_sae_module(sae_src: str):
    """Import the ``ml_service`` ``sae.py`` from an arbitrary file path.

    Uses ``importlib`` (not ``sys.path`` mutation) so the import is hermetic and
    leaves the interpreter's import state untouched.
    """
    spec = importlib.util.spec_from_file_location("vllm_sae_scorer_sae", sae_src)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import SAE module from {sae_src!r}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's own globals (notably ``torch``) are
    # resolvable as ``sys.modules[cls.__module__].__dict__`` — pydantic needs
    # this to resolve the ``torch.dtype`` forward ref on ``SAESignalConfig``
    # (the SAE module uses ``from __future__ import annotations``).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_sae(
    sae_src: str,
    sae_dir: str,
    signal_name: str | None,
    device: torch.device | str,
):
    """Load an ``SAESignal`` from ``sae_dir`` and move it to ``device``.

    Factored out as a module-level function so tests can monkeypatch it with a
    small in-memory SAE instead of touching disk.
    """
    module = _import_sae_module(sae_src)
    sae = module.load_sae_local(sae_dir, signal_name)
    sae.eval()
    sae.to(device)
    return sae


class SaeScorerConsumer(CaptureConsumer):
    """Score a single layer's residual stream with an SAE, on the GPU."""

    location: Literal["worker", "driver"] = "worker"
    wants_device_tensors: ClassVar[bool] = True

    def __init__(self, vllm_config: VllmConfig, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self._layer = int(params.get("layer", DEFAULT_LAYER))
        self._hook = str(params.get("hook", DEFAULT_HOOK))
        self._positions = params.get("positions", DEFAULT_POSITIONS)
        # ``topk`` (optional): keep only the k strongest features per row.
        # ``None`` (default) keeps every nonzero feature (true JumpReLU sparsity).
        topk = params.get("topk")
        self._topk: int | None = int(topk) if topk is not None else None
        # ``skip_first``: emit no features for row 0 (the first captured position
        # — BOS / attention-sink), whose SAE activation is anomalously dense
        # (~half the dictionary) and uninformative. Skips its nonzero/tolist (the
        # dominant per-request egress cost) while keeping the row for alignment.
        self._skip_first: bool = bool(params.get("skip_first", False))

        # Store mode (solution 2): write scores to disk under ``out_dir`` and
        # return a handle. ``None`` keeps inline mode (return full scores).
        out_dir = params.get("out_dir")
        self._out_dir: pathlib.Path | None = (
            pathlib.Path(out_dir).expanduser() if out_dir else None
        )

        device = params.get("device") or self._resolve_device(vllm_config)
        self._sae = load_sae(
            sae_src=str(params.get("sae_src", DEFAULT_SAE_SRC)),
            sae_dir=str(params.get("sae_dir", DEFAULT_SAE_DIR)),
            signal_name=params.get("signal_name"),
            device=device,
        )
        self._sae_size = int(self._sae.config.sae_size)

    @staticmethod
    def _resolve_device(vllm_config: VllmConfig) -> torch.device | str:
        device_config = getattr(vllm_config, "device_config", None)
        device = getattr(device_config, "device", None)
        if device is not None:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(
            hooks={self._hook: [self._layer]},
            positions=self._positions,
        )

    @torch.inference_mode()
    def _score(self, tensor: torch.Tensor) -> dict[str, Any]:
        """Encode the captured rows into a sparse-scores dict.

        ``tensor`` is ``(num_rows, hidden_size)`` on the model device. Rows are
        in ascending captured-position order (the manager delivers chunks in
        step order; the adapter concatenates them in that order).
        """
        num_rows = int(tensor.shape[0]) if tensor.ndim > 0 else 0
        scores: dict[str, Any] = {
            "layer": self._layer,
            "hook": self._hook,
            "sae_size": self._sae_size,
            "num_rows": num_rows,
            "indices": [],
            "values": [],
        }
        if num_rows == 0:
            return scores

        # encode casts to the SAE's dtype internally; runs on ``tensor``'s
        # device (the GPU, for a device-resident delivery).
        feats = self._sae.encode(tensor)  # (num_rows, sae_size)

        indices: list[list[int]] = []
        values: list[list[float]] = []
        for i, row in enumerate(feats):
            if self._skip_first and i == 0:
                # BOS / first-position row: dense artifact — emit empty, skip the
                # costly sparsify, keep the row so downstream alignment holds.
                indices.append([])
                values.append([])
                continue
            if self._topk is not None:
                k = min(self._topk, row.numel())
                vals, idx = torch.topk(row, k)
                # Drop any zeros pulled in when fewer than k features are active.
                keep = vals > 0
                idx, vals = idx[keep], vals[keep]
            else:
                idx = row.nonzero(as_tuple=True)[0]
                vals = row[idx]
            indices.append(idx.detach().to("cpu", torch.int64).tolist())
            values.append(vals.detach().to("cpu", torch.float32).tolist())

        scores["indices"] = indices
        scores["values"] = values
        return scores

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> dict[str, Any]:
        """Score the captured rows; store to disk + return a handle, or inline.

        In store mode (``out_dir`` set) the full scores are written atomically to
        :func:`score_path` and only a small handle rides ``capture_results``; the
        on-disk file is the source of truth and is retrievable by request id
        regardless of capture-result delivery. In inline mode the full sparse
        scores are returned directly.
        """
        scores = self._score(tensor)
        if self._out_dir is None:
            return scores

        request_id = str(key[0])
        path = score_path(self._out_dir, request_id, self._layer, self._hook)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic publish: write to a temp file in the same dir, then rename, so a
        # reader never observes a partially written file.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(scores))
        os.replace(tmp, path)

        return {
            "stored": True,
            "path": str(path),
            "request_id": request_id,
            "layer": self._layer,
            "hook": self._hook,
            "sae_size": self._sae_size,
            "num_rows": scores["num_rows"],
        }
