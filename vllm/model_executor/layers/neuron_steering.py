# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request neuron steering on the MLP intermediate (the input of ``down_proj``).

Each steerable MLP carries four small tables of shape ``(num_presets + 1, K)``
-- neuron index, orientation ``s``, threshold ``theta``, coefficient ``c`` --
plus a per-token row index shared by all steerable MLPs (row 0 = no steering,
row ``i`` = preset ``NeuronSteeringConfig.names[i - 1]``). The model runner
writes the row index before each forward. For every token::

    h[n] <- h[n] + c * s * relu(s * (h[n] - theta))      for the K slots of its row

computed in float32 and cast back, matching the HF reference hooks. Row 0 and
padding slots have ``c = 0`` and point at distinct neurons, so they write each
value back unchanged and the scatter never writes one position twice.

The op is a plain functional custom op (no host sync, no data-dependent
shapes), so it is captured inside both piecewise and full CUDA graphs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.neuron_steering import NeuronSteeringConfig

NSTEER_ATTRS = ("nsteer_idx", "nsteer_s", "nsteer_theta", "nsteer_c")
NSTEER_INDEX_ATTR = "nsteer_index"


def neuron_steering_config(vllm_config: VllmConfig) -> NeuronSteeringConfig | None:
    return getattr(vllm_config, "neuron_steering_config", None)


def build_layer_tables(cfg: NeuronSteeringConfig, layer_idx: int, intermediate_size: int):
    """``(idx, s, theta, c)`` CPU tables for one layer, rows = [none] + presets."""
    names, k = cfg.names, cfg.max_neurons_per_layer
    rows = len(names) + 1
    idx = torch.zeros(rows, k, dtype=torch.long)
    s = torch.ones(rows, k, dtype=torch.float32)
    theta = torch.zeros(rows, k, dtype=torch.float32)
    c = torch.zeros(rows, k, dtype=torch.float32)
    idx[0] = torch.arange(k)
    for r, name in enumerate(names, start=1):
        entries = cfg.presets[name].get(layer_idx, [])
        used = {n for n, *_ in entries}
        for n, *_ in entries:
            if not 0 <= n < intermediate_size:
                raise ValueError(f"neuron steering: preset {name!r} layer {layer_idx}: neuron {n} out of range")
        pad = (p for p in range(intermediate_size) if p not in used)
        for j in range(k):
            if j < len(entries):
                n, sj, tj, cj = entries[j]
                idx[r, j], s[r, j], theta[r, j], c[r, j] = n, sj, tj, cj
            else:
                idx[r, j] = next(pad)  # distinct, untouched neuron; c = 0 writes it back unchanged
    return idx, s, theta, c


def register_mlp_neuron_steering(
    mlp: nn.Module, cfg: NeuronSteeringConfig | None, layer_idx: int, intermediate_size: int
) -> bool:
    """Attach the tables to ``mlp`` if any preset steers ``layer_idx``. Returns whether it did."""
    if cfg is None or layer_idx not in cfg.layers:
        return False
    from vllm.distributed import get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() != 1:
        raise ValueError("neuron steering needs tensor_parallel_size == 1 (neuron ids index the full intermediate)")
    for attr, t in zip(NSTEER_ATTRS, build_layer_tables(cfg, layer_idx, intermediate_size)):
        mlp.register_buffer(attr, t, persistent=False)
    return True


def share_neuron_steering_index(mlps: list[nn.Module], max_tokens: int) -> torch.Tensor | None:
    """One ``(max_tokens,)`` row-index tensor shared by every steerable MLP."""
    steered = [m for m in mlps if hasattr(m, NSTEER_ATTRS[0])]
    if not steered:
        return None
    index = torch.zeros(max_tokens, dtype=torch.long, device=getattr(steered[0], NSTEER_ATTRS[0]).device)
    for m in steered:
        m.register_buffer(NSTEER_INDEX_ATTR, index, persistent=False)
    return index


def maybe_apply_mlp_neuron_steering(mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Static per-module branch: MLPs without tables emit no op at all."""
    if not hasattr(mlp, NSTEER_INDEX_ATTR):
        return x
    return torch.ops.vllm.apply_neuron_steering(
        x, mlp.nsteer_idx, mlp.nsteer_s, mlp.nsteer_theta, mlp.nsteer_c, mlp.nsteer_index
    )


def apply_neuron_steering(
    x: torch.Tensor,
    idx: torch.Tensor,
    s: torch.Tensor,
    theta: torch.Tensor,
    c: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    r = index[: x.shape[0]]
    cols = idx[r]  # (T, K)
    h = torch.gather(x, 1, cols).float()
    sr = s[r]
    new = h + c[r] * sr * torch.relu(sr * (h - theta[r]))
    return x.scatter(1, cols, new.to(x.dtype))


def apply_neuron_steering_fake(x, idx, s, theta, c, index) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="apply_neuron_steering",
    op_func=apply_neuron_steering,
    fake_impl=apply_neuron_steering_fake,
    mutates_args=[],
)
