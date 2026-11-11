# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request neuron steering on the MLP intermediate (the input of ``down_proj``).

Each steerable MLP carries fixed tables for the ranked neurons in its layer (from
``NeuronSteeringConfig``): neuron index ``idx (K,)``, orientation ``s (K,)``, global rank
``rank (K,)`` and thresholds ``theta (101, K)`` by percentile index. Three per-token vectors are
shared by every steerable MLP and written by the model runner before each forward: ``k`` (how
many of the ranking's top neurons the token's request steers; 0 = none), ``q`` (percentile
index) and ``c`` (coefficient). For every token and every neuron of the layer::

    active = rank < k
    h[n] <- h[n] + (c * active) * s * relu(s * (h[n] - theta[q]))

computed in float32 and cast back, matching the HF reference hooks. An inactive neuron or an
unsteered token (``k = 0``, ``c = 0``) gets ``h + 0``: its value is written back unchanged.

Only the per-token vectors depend on what requests send; the tables and shapes are fixed by the
ranking, so the op is a plain functional custom op (no host sync, no data-dependent shapes)
captured inside both piecewise and full CUDA graphs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.neuron_steering import NeuronSteeringConfig

NSTEER_ATTRS = ("nsteer_idx", "nsteer_s", "nsteer_rank", "nsteer_theta")
NSTEER_TOKEN_ATTRS = ("nsteer_k", "nsteer_q", "nsteer_c")


def neuron_steering_config(vllm_config: VllmConfig) -> NeuronSteeringConfig | None:
    return getattr(vllm_config, "neuron_steering_config", None)


def build_layer_tables(cfg: NeuronSteeringConfig, layer_idx: int, intermediate_size: int):
    """``(idx, s, rank, theta)`` CPU tables for the ranked neurons of one layer."""
    entries = cfg.layer_entries(layer_idx)
    for _rank, n, _s, _t in entries:
        if not 0 <= n < intermediate_size:
            raise ValueError(f"neuron steering: layer {layer_idx}: neuron {n} out of range")
    idx = torch.tensor([n for _, n, _, _ in entries], dtype=torch.long)
    s = torch.tensor([sj for _, _, sj, _ in entries], dtype=torch.float32)
    rank = torch.tensor([r for r, _, _, _ in entries], dtype=torch.long)
    theta = torch.tensor([t for _, _, _, t in entries], dtype=torch.float32).T.contiguous()  # (101, K)
    return idx, s, rank, theta


def register_mlp_neuron_steering(
    mlp: nn.Module, cfg: NeuronSteeringConfig | None, layer_idx: int, intermediate_size: int
) -> bool:
    """Attach the tables to ``mlp`` if the ranking has neurons in ``layer_idx``. Returns whether it did."""
    if cfg is None or layer_idx not in cfg.layers:
        return False
    from vllm.distributed import get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() != 1:
        raise ValueError("neuron steering needs tensor_parallel_size == 1 (neuron ids index the full intermediate)")
    for attr, t in zip(NSTEER_ATTRS, build_layer_tables(cfg, layer_idx, intermediate_size)):
        mlp.register_buffer(attr, t, persistent=False)
    return True


def share_neuron_steering_index(
    mlps: list[nn.Module], max_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """The per-token ``(k, q, c)`` vectors, ``(max_tokens,)`` each, shared by every steerable MLP."""
    steered = [m for m in mlps if hasattr(m, NSTEER_ATTRS[0])]
    if not steered:
        return None
    dev = getattr(steered[0], NSTEER_ATTRS[0]).device
    vecs = (
        torch.zeros(max_tokens, dtype=torch.long, device=dev),
        torch.zeros(max_tokens, dtype=torch.long, device=dev),
        torch.zeros(max_tokens, dtype=torch.float32, device=dev),
    )
    for m in steered:
        for attr, v in zip(NSTEER_TOKEN_ATTRS, vecs):
            m.register_buffer(attr, v, persistent=False)
    return vecs


def maybe_apply_mlp_neuron_steering(mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Static per-module branch: MLPs without tables emit no op at all."""
    if not hasattr(mlp, NSTEER_TOKEN_ATTRS[0]):
        return x
    return torch.ops.vllm.apply_neuron_steering(
        x, mlp.nsteer_idx, mlp.nsteer_s, mlp.nsteer_rank, mlp.nsteer_theta,
        mlp.nsteer_k, mlp.nsteer_q, mlp.nsteer_c,
    )


def apply_neuron_steering(
    x: torch.Tensor,
    idx: torch.Tensor,
    s: torch.Tensor,
    rank: torch.Tensor,
    theta: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    t = x.shape[0]
    h = x.index_select(1, idx).float()                                   # (T, K)
    coef = c[:t, None] * (rank[None, :] < k[:t, None])                   # (T, K)
    new = h + coef * s * torch.relu(s * (h - theta[q[:t]]))
    return x.index_copy(1, idx, new.to(x.dtype))


def apply_neuron_steering_fake(x, idx, s, rank, theta, k, q, c) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="apply_neuron_steering",
    op_func=apply_neuron_steering,
    fake_impl=apply_neuron_steering_fake,
    mutates_args=[],
)
