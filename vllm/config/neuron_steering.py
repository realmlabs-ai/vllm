# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named neuron-steering presets (per-request, MLP-intermediate transport maps).

A preset rewrites a few MLP neurons -- the inputs ``h`` of ``mlp.down_proj`` --
at every token of the requests that select it::

    h <- h + c * s * relu(s * (h - theta))

per neuron, with orientation ``s`` (+1/-1), threshold ``theta`` and coefficient
``c``. ``c = -(1 + alpha)`` reflects the excess past ``theta`` (Veer's
``reflect``, away from the behaviour); ``c = gamma`` amplifies it
(``amplify``, toward). Presets are fixed at engine start (loaded from a JSON
file), so requests can only pick a name: they cannot send arbitrary
interventions.

File format::

    {"presets": {"<name>": {"layers": {"<layer>": [[neuron, s, theta, c], ...]}}}}

When no presets are configured the engine registers no buffers and the model
emits no steering op, so the forward pass is unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ConfigDict, Field

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash


def _validate(presets: dict[str, Any]) -> dict[str, dict[int, list[tuple[int, float, float, float]]]]:
    out: dict[str, dict[int, list[tuple[int, float, float, float]]]] = {}
    if not isinstance(presets, dict) or not presets:
        raise ValueError("neuron steering: 'presets' must be a non-empty object")
    for name, p in presets.items():
        if not isinstance(name, str) or not name:
            raise ValueError("neuron steering: preset names must be non-empty strings")
        layers = (p or {}).get("layers")
        if not isinstance(layers, dict) or not layers:
            raise ValueError(f"neuron steering: preset {name!r} has no layers")
        out[name] = {}
        for layer, entries in layers.items():
            rows = []
            for e in entries:
                if len(e) != 4:
                    raise ValueError(f"neuron steering: preset {name!r} layer {layer}: entries are [neuron, s, theta, c]")
                n, s, theta, c = int(e[0]), float(e[1]), float(e[2]), float(e[3])
                if s not in (1.0, -1.0):
                    raise ValueError(f"neuron steering: preset {name!r}: s must be +1 or -1")
                rows.append((n, s, theta, c))
            if len({r[0] for r in rows}) != len(rows):
                raise ValueError(f"neuron steering: preset {name!r} layer {layer}: duplicate neuron")
            out[name][int(layer)] = rows
    return out


@config(config=ConfigDict(arbitrary_types_allowed=True))
class NeuronSteeringConfig:
    """Per-request neuron steering from named presets."""

    presets: dict[str, dict[int, list[tuple[int, float, float, float]]]] = Field(default_factory=dict)
    """``name -> layer -> [(neuron, s, theta, c), ...]``."""

    @classmethod
    def from_file(cls, path: str) -> NeuronSteeringConfig:
        with open(path) as fh:
            doc = json.load(fh)
        return cls(presets=_validate(doc.get("presets")))

    @property
    def names(self) -> list[str]:
        """Preset names in table-row order (row = index + 1; row 0 = no steering)."""
        return sorted(self.presets)

    @property
    def layers(self) -> list[int]:
        return sorted({layer for p in self.presets.values() for layer in p})

    @property
    def max_neurons_per_layer(self) -> int:
        return max(len(rows) for p in self.presets.values() for rows in p.values())

    def row_of(self, name: str | None) -> int:
        if name is None:
            return 0
        try:
            return self.names.index(name) + 1
        except ValueError:
            raise ValueError(
                f"unknown neuron_steering preset {name!r}; available: {self.names}"
            ) from None

    def compute_hash(self) -> str:
        # The presets set which layers carry the op and the table shapes, so
        # they are part of the compiled graph's identity.
        factors = [sorted((n, sorted((k, v) for k, v in p.items())) for n, p in self.presets.items())]
        return safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
