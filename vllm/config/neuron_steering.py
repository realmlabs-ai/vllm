# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request neuron steering from a fixed neuron ranking (MLP-intermediate transport maps).

The engine loads a ranking at start (Veer's ``steer.json``): neurons in rank order, each a
``(layer, neuron)`` of the inputs ``h`` of ``mlp.down_proj``, with an orientation ``s`` (+1/-1)
and 101 activation percentiles ``qA``. A request steers with a spec string::

    <reflect|amplify>:top<k>:<q>:<strength>:all

which rewrites the top ``k`` neurons of the ranking at every token of that request::

    h <- h + c * s * relu(s * (h - theta))

with ``theta = qA[j][round(100 q)]`` if ``s > 0`` else ``qA[j][100 - round(100 q)]``, and
``c = -(1 + strength)`` (reflect: away from the behaviour) or ``c = strength`` (amplify: toward).
This mirrors Veer's ``cpc_space/app.py``. Any ``k`` up to the ranking's length, any ``q`` in
[0, 1] and any finite ``strength >= 0`` is accepted; the neurons and layers are fixed by the file,
so the model's tables and compiled graphs do not depend on what requests send.

File format (``steer.json``)::

    {"model": "...", "neurons": [[layer, neuron], ...], "s": [1, -1, ...], "qA": [[101 floats], ...]}

When no ranking is configured the engine registers no buffers and the model emits no steering op.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from pydantic import ConfigDict, Field

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

SPEC_RE = re.compile(r"(reflect|amplify):top(\d+):(\d+(?:\.\d+)?|\.\d+):(\d+(?:\.\d+)?|\.\d+):all")
NUM_PERCENTILES = 101


def _validate(doc: Any) -> tuple[list[tuple[int, int]], list[float], list[list[float]]]:
    if not isinstance(doc, dict):
        raise ValueError("neuron steering: the ranking file must be a JSON object")
    neurons, s, qa = doc.get("neurons"), doc.get("s"), doc.get("qA")
    if not isinstance(neurons, list) or not neurons:
        raise ValueError("neuron steering: 'neurons' must be a non-empty list of [layer, neuron]")
    if not isinstance(s, list) or not isinstance(qa, list) or not len(neurons) == len(s) == len(qa):
        raise ValueError("neuron steering: 'neurons', 's' and 'qA' must be lists of the same length")
    out_n: list[tuple[int, int]] = []
    for e in neurons:
        if not isinstance(e, list) or len(e) != 2:
            raise ValueError("neuron steering: each entry of 'neurons' is [layer, neuron]")
        out_n.append((int(e[0]), int(e[1])))
    if len(set(out_n)) != len(out_n):
        raise ValueError("neuron steering: a neuron appears twice in the ranking")
    out_s = [float(v) for v in s]
    if any(v not in (1.0, -1.0) for v in out_s):
        raise ValueError("neuron steering: every 's' must be +1 or -1")
    out_q: list[list[float]] = []
    for row in qa:
        if not isinstance(row, list) or len(row) != NUM_PERCENTILES:
            raise ValueError(f"neuron steering: every 'qA' row needs {NUM_PERCENTILES} percentiles")
        vals = [float(v) for v in row]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("neuron steering: 'qA' values must be finite")
        out_q.append(vals)
    return out_n, out_s, out_q


@config(config=ConfigDict(arbitrary_types_allowed=True))
class NeuronSteeringConfig:
    """Per-request neuron steering over a fixed neuron ranking."""

    neurons: list[tuple[int, int]] = Field(default_factory=list)
    """``(layer, neuron)`` in rank order."""
    s: list[float] = Field(default_factory=list)
    """Orientation per ranked neuron, +1 or -1."""
    qa: list[list[float]] = Field(default_factory=list)
    """101 activation percentiles per ranked neuron."""

    @classmethod
    def from_file(cls, path: str) -> NeuronSteeringConfig:
        with open(path) as fh:
            doc = json.load(fh)
        n, s, qa = _validate(doc)
        return cls(neurons=n, s=s, qa=qa)

    @property
    def layers(self) -> list[int]:
        return sorted({layer for layer, _ in self.neurons})

    def layer_entries(self, layer: int) -> list[tuple[int, int, float, list[float]]]:
        """``(rank, neuron, s, theta by percentile index)`` for the ranked neurons in ``layer``.

        ``theta[q]`` already applies the orientation: ``qA[q]`` if ``s > 0`` else ``qA[100 - q]``.
        """
        out = []
        for j, (lay, neuron) in enumerate(self.neurons):
            if lay != layer:
                continue
            sj, row = self.s[j], self.qa[j]
            theta = row if sj > 0 else row[::-1]
            out.append((j, neuron, sj, list(theta)))
        return out

    def parse(self, spec: str) -> tuple[int, int, float]:
        """``spec`` -> ``(k, percentile index, c)``; raises ``ValueError`` on anything else."""
        m = SPEC_RE.fullmatch(spec.strip()) if isinstance(spec, str) else None
        if m is None:
            raise ValueError(
                f"neuron_steering {spec!r}: expected '<reflect|amplify>:top<k>:<q>:<strength>:all'"
            )
        fam, k, q, strength = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
        if not 1 <= k <= len(self.neurons):
            raise ValueError(f"neuron_steering {spec!r}: k must be 1..{len(self.neurons)}")
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"neuron_steering {spec!r}: q must be in [0, 1]")
        if not math.isfinite(strength):
            raise ValueError(f"neuron_steering {spec!r}: strength must be finite")
        c = -(1.0 + strength) if fam == "reflect" else strength
        return k, round(q * 100), c

    def compute_hash(self) -> str:
        # The ranking sets which layers carry the op and the table shapes, so it is
        # part of the compiled graph's identity. Request specs are not.
        factors = [self.neurons, self.s, self.qa]
        return safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
