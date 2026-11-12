# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission-time validation for filesystem capture requests.

Runs at the OpenAI-compatible entrypoint boundary (or at
``LLM(...)`` admission) before the request is handed to the engine.
Takes a :class:`FilesystemCaptureRequest` + :class:`VllmConfig` +
:class:`CaptureContext` and either returns a :class:`CaptureSpec`
that downstream code can consume, or raises
:class:`CaptureValidationError` with a descriptive message that the
entrypoint surfaces as HTTP 400.

The checks performed here are the ones that can't be done at
``SamplingParams`` construction time because they need a
``VllmConfig`` or request context:

1. Every layer referenced by every hook is in
   ``[0, num_hidden_layers)`` (the *global* layer count under PP).
2. Tag and request_id slug cleanly (reject ``..``, leading ``/``,
   > 256 chars, empty).
3. Explicit position indices are valid and not below
   ``num_computed_tokens`` (prefix-cache hits that were never
   forwarded).

Tensor / pipeline / expert / data parallelism are all supported for
the replicated residual hooks; see the parallelism note at the
rejection site (now removed) and ``docs/design/capture_parallelism.md``.

Runtime concerns (writer pool, plan building, finalization) are not
this module's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import regex as re

from vllm.v1.capture.errors import CaptureValidationError
from vllm.v1.capture.types import (
    CaptureContext,
    CaptureSpec,
    HookName,
    LayerMix,
    PositionSelector,
)

from .types import FilesystemCaptureRequest

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# ---------------------------------------------------------------------------
# Hook-point + position-kind literal sets
# ---------------------------------------------------------------------------

# Hook names mirror ``vllm.model_executor.layers.steering``; redeclared
# here (rather than imported) to keep this module torch-free. Whenever
# steering grows a new hook point, this tuple must be updated in
# lockstep.
#
# ``mlp_in`` / ``mlp_out`` are reserved in the hook-id table
# (``activation_capture.py``) and ``HookName`` but are not yet wired into
# any model forward, so a request for them would pass validation and then
# produce an empty, zero-byte capture. Keep them out of the accepted set
# so admission rejects them until they are wired; re-add here once
# implemented.
#
# ``post_norm`` IS wired (llama / qwen2 / qwen3_next stack forwards) but fires
# only at ``layer_idx = num_hidden_layers - 1`` -- it taps the stack-final norm,
# not a per-layer point. Requesting it at any other layer captures nothing.
_VALID_HOOK_NAMES: frozenset[str] = frozenset(
    ("pre_attn", "post_attn", "post_block", "post_norm", "mlp_out", "layer_mix")
)
# Hooks a ``layer_mix`` member may name: real forward taps only.
_MIX_MEMBER_HOOKS: frozenset[str] = frozenset(
    ("pre_attn", "post_attn", "mlp_out", "post_block", "post_norm")
)

_VALID_POSITION_KINDS: frozenset[str] = frozenset(
    ("last_prompt", "all_prompt", "all_generated", "all")
)


# ---------------------------------------------------------------------------
# Slugging
# ---------------------------------------------------------------------------

_SLUG_REGEX = re.compile(r"[^a-zA-Z0-9._-]")
_SLUG_MAX_LEN = 256


def _slug(name: str, *, field: str) -> str:
    """Slug a ``tag`` or ``request_id`` into a safe path segment.

    Rejects empty input, ``..`` (path traversal), leading ``/``, and
    anything over 256 chars.
    """
    if not isinstance(name, str):
        raise CaptureValidationError(
            f"{field} must be a string, got {type(name).__name__}"
        )
    if not name:
        raise CaptureValidationError(f"{field} must be non-empty")
    if len(name) > _SLUG_MAX_LEN:
        raise CaptureValidationError(
            f"{field} must be at most {_SLUG_MAX_LEN} characters, got {len(name)}"
        )
    if ".." in name:
        raise CaptureValidationError(f"{field} must not contain '..': {name!r}")
    if name.startswith("/"):
        raise CaptureValidationError(f"{field} must not start with '/': {name!r}")
    return _SLUG_REGEX.sub("_", name)


# ---------------------------------------------------------------------------
# Layer selector expansion
# ---------------------------------------------------------------------------


def _expand_layer_list(
    raw: list[int], *, num_hidden_layers: int, where: str
) -> list[int]:
    result: list[int] = []
    for i, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CaptureValidationError(
                f"{where}[{i}] must be an int, got {type(value).__name__}"
            )
        if value < 0 or value >= num_hidden_layers:
            raise CaptureValidationError(
                f"{where}[{i}] = {value} is out of range for a model with "
                f"num_hidden_layers={num_hidden_layers}"
            )
        result.append(value)
    return result


def _expand_ranges(raw: list[Any], *, num_hidden_layers: int, where: str) -> list[int]:
    result: list[int] = []
    for i, pair in enumerate(raw):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise CaptureValidationError(
                f"{where}[{i}] must be a 2-element [start, end] pair, got {pair!r}"
            )
        start, end = pair
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise CaptureValidationError(
                f"{where}[{i}] must be a pair of ints, got {pair!r}"
            )
        if start > end:
            raise CaptureValidationError(
                f"{where}[{i}] has start > end ({start} > {end})"
            )
        if start < 0 or end >= num_hidden_layers:
            raise CaptureValidationError(
                f"{where}[{i}] = [{start}, {end}] is out of range for a "
                f"model with num_hidden_layers={num_hidden_layers}"
            )
        result.extend(range(start, end + 1))
    return result


def _expand_hook_layers(
    selector: Any, num_hidden_layers: int, *, where: str
) -> list[int]:
    """Resolve a layer selector into a sorted, deduped list of indices.

    Accepts ``"all"``, ``list[int]``, or ``{"layers": [...], "ranges": [[a,b], ...]}``.
    """
    if num_hidden_layers <= 0:
        raise CaptureValidationError(
            f"num_hidden_layers must be positive, got {num_hidden_layers}"
        )

    if isinstance(selector, str):
        if selector != "all":
            raise CaptureValidationError(
                f"{where} string form must be 'all', got {selector!r}"
            )
        return list(range(num_hidden_layers))

    if isinstance(selector, list):
        resolved = _expand_layer_list(
            selector, num_hidden_layers=num_hidden_layers, where=where
        )
        return sorted(set(resolved))

    if isinstance(selector, dict):
        allowed = {"layers", "ranges"}
        extra = set(selector.keys()) - allowed
        if extra:
            raise CaptureValidationError(
                f"{where} dict form has unexpected keys {sorted(extra)}; "
                f"allowed keys: {sorted(allowed)}"
            )
        layers_raw = selector.get("layers")
        ranges_raw = selector.get("ranges")
        if layers_raw is None and ranges_raw is None:
            raise CaptureValidationError(
                f"{where} dict form must set 'layers' and/or 'ranges'; got empty"
            )

        collected: list[int] = []
        if layers_raw is not None:
            if not isinstance(layers_raw, list):
                raise CaptureValidationError(
                    f"{where}.layers must be a list of ints, got "
                    f"{type(layers_raw).__name__}"
                )
            collected.extend(
                _expand_layer_list(
                    layers_raw,
                    num_hidden_layers=num_hidden_layers,
                    where=f"{where}.layers",
                )
            )
        if ranges_raw is not None:
            if not isinstance(ranges_raw, list):
                raise CaptureValidationError(
                    f"{where}.ranges must be a list of [start, end] pairs, "
                    f"got {type(ranges_raw).__name__}"
                )
            collected.extend(
                _expand_ranges(
                    ranges_raw,
                    num_hidden_layers=num_hidden_layers,
                    where=f"{where}.ranges",
                )
            )
        return sorted(set(collected))

    raise CaptureValidationError(
        f"{where} must be 'all', list[int], or {{'layers': ..., 'ranges': ...}}; "
        f"got {type(selector).__name__}"
    )


# ---------------------------------------------------------------------------
# Position selector resolution
# ---------------------------------------------------------------------------


def _resolve_positions(
    selector: Any,
    num_prompt_tokens: int,
    num_generated_tokens: int,
    *,
    where: str,
) -> list[int]:
    """Resolve a position selector into absolute token indices.

    Only handles the cases whose size is known from static request
    metadata. ``all_generated`` and ``all`` are symbolic — callers
    should defer those to the runner.
    """
    if num_prompt_tokens <= 0:
        raise CaptureValidationError(
            f"num_prompt_tokens must be positive, got {num_prompt_tokens}"
        )
    if num_generated_tokens < 0:
        raise CaptureValidationError(
            f"num_generated_tokens must be non-negative, got {num_generated_tokens}"
        )

    if isinstance(selector, str):
        if selector not in _VALID_POSITION_KINDS:
            raise CaptureValidationError(
                f"{where} must be one of {sorted(_VALID_POSITION_KINDS)} "
                f"or a list of ints, got {selector!r}"
            )
        if selector == "last_prompt":
            return [num_prompt_tokens - 1]
        if selector == "all_prompt":
            return list(range(0, num_prompt_tokens))
        if selector == "all_generated":
            return list(
                range(num_prompt_tokens, num_prompt_tokens + num_generated_tokens)
            )
        # "all"
        return list(range(0, num_prompt_tokens + num_generated_tokens))

    if isinstance(selector, list):
        total = num_prompt_tokens + num_generated_tokens
        result: list[int] = []
        for i, value in enumerate(selector):
            if isinstance(value, bool) or not isinstance(value, int):
                raise CaptureValidationError(
                    f"{where}[{i}] must be an int, got {type(value).__name__}"
                )
            if value < 0 or value >= total:
                raise CaptureValidationError(
                    f"{where}[{i}] = {value} is out of range for a request "
                    f"with num_prompt_tokens={num_prompt_tokens}, "
                    f"num_generated_tokens={num_generated_tokens}"
                )
            result.append(value)
        return sorted(set(result))

    raise CaptureValidationError(
        f"{where} must be one of {sorted(_VALID_POSITION_KINDS)} or list[int], "
        f"got {type(selector).__name__}"
    )


# ---------------------------------------------------------------------------
# Public API: validate_filesystem_request
# ---------------------------------------------------------------------------


def _structural_validate(raw: FilesystemCaptureRequest) -> None:
    """Structural validation of the raw request before resolving."""
    if not isinstance(raw.request_id, str) or not raw.request_id:
        raise CaptureValidationError(
            f"capture.request_id must be a non-empty string, got {raw.request_id!r}"
        )
    if not isinstance(raw.tag, str) or not raw.tag:
        raise CaptureValidationError(
            f"capture.tag must be a non-empty string, got {raw.tag!r}"
        )
    if not isinstance(raw.hooks, dict) or not raw.hooks:
        raise CaptureValidationError(
            "capture.hooks must be a non-empty dict mapping hook point "
            "name to a layer selector"
        )
    for hook_name in raw.hooks:
        if not isinstance(hook_name, str):
            raise CaptureValidationError(
                f"capture.hooks key must be a string, got {type(hook_name).__name__}"
            )
        if hook_name not in _VALID_HOOK_NAMES:
            raise CaptureValidationError(
                f"capture.hooks key {hook_name!r} is not a valid hook "
                f"point; valid names: {sorted(_VALID_HOOK_NAMES)}"
            )
    if isinstance(raw.positions, str):
        if raw.positions not in _VALID_POSITION_KINDS:
            raise CaptureValidationError(
                f"capture.positions string form must be one of "
                f"{sorted(_VALID_POSITION_KINDS)}, got {raw.positions!r}"
            )
    elif isinstance(raw.positions, list):
        if not raw.positions:
            raise CaptureValidationError(
                "capture.positions list form must be non-empty"
            )
        for i, value in enumerate(raw.positions):
            if isinstance(value, bool) or not isinstance(value, int):
                raise CaptureValidationError(
                    f"capture.positions[{i}] must be an int, got {type(value).__name__}"
                )
            if value < 0:
                raise CaptureValidationError(
                    f"capture.positions[{i}] = {value} must be non-negative"
                )
    else:
        raise CaptureValidationError(
            "capture.positions must be a string "
            f"({sorted(_VALID_POSITION_KINDS)}) or a list of ints, "
            f"got {type(raw.positions).__name__}"
        )


def validate_filesystem_request(
    raw: FilesystemCaptureRequest,
    vllm_config: VllmConfig,
    ctx: CaptureContext,
) -> CaptureSpec:
    """Validate a ``FilesystemCaptureRequest`` and return a ``CaptureSpec``.

    Raises :class:`CaptureValidationError` on any validation failure.
    The returned spec's ``positions`` is either a resolved ``list[int]``
    (``last_prompt`` / ``all_prompt`` / explicit list) or a symbolic
    kind string (``all_generated`` / ``all``) which the runner
    materializes per-step.
    """
    # 1. Structural validation.
    _structural_validate(raw)

    # 2. Parallelism. The residual hooks captured today (pre_attn /
    # post_attn / post_block) read the residual stream after the
    # tensor-parallel all-reduce / MoE combine, so it is replicated and
    # full-width across the TP and EP planes; data parallelism partitions
    # requests across independent engine cores. All four axes are
    # therefore supported: TP rank 0 of each pipeline stage captures that
    # stage's (global-indexed) layers to the shared mount, and the engine
    # merges the per-stage results. No accept/reject branch on TP / PP /
    # EP / DP size is needed here.
    #
    # Phase 4 (sharded-activation capture: MLP intermediate / per-expert
    # outputs) will reintroduce a rejection *for sharded hooks only*,
    # since those tensors are partitioned across the TP / EP plane and
    # need a gather. The residual hooks above remain unconditional.

    # 3. Slug tag + request_id so we surface a clean error (rather
    # than failing much later on a filesystem write).
    _slug(raw.request_id, field="capture.request_id")
    _slug(raw.tag, field="capture.tag")

    # 4. Expand hooks and validate layer indices.
    if ctx.num_hidden_layers <= 0:
        raise CaptureValidationError(
            f"filesystem capture requires num_hidden_layers > 0; got "
            f"{ctx.num_hidden_layers}"
        )
    resolved_hooks: dict[HookName, list[int]] = {}
    for hook_name, selector in raw.hooks.items():
        resolved = _expand_hook_layers(
            selector,
            ctx.num_hidden_layers,
            where=f"capture.hooks[{hook_name!r}]",
        )
        if not resolved:
            raise CaptureValidationError(
                f"capture.hooks[{hook_name!r}] expanded to an empty layer list"
            )
        # hook_name already validated as one of _VALID_HOOK_NAMES.
        resolved_hooks[cast(HookName, hook_name)] = resolved

    # 5. Resolve positions.
    # "last_prompt", "all_prompt", and explicit lists resolve at
    # admission time. "all_generated" and "all" stay symbolic — the
    # runner materializes them per-step once the generated token
    # count is known.
    resolved_positions: PositionSelector
    if isinstance(raw.positions, str):
        if raw.positions in ("last_prompt", "all_prompt"):
            resolved_positions = _resolve_positions(
                raw.positions,
                ctx.num_prompt_tokens,
                num_generated_tokens=0,
                where="capture.positions",
            )
        else:
            # "all_generated" / "all" — defer to the runner. Already
            # validated against _VALID_POSITION_KINDS above.
            resolved_positions = cast(PositionSelector, raw.positions)
    elif isinstance(raw.positions, list):
        resolved_positions = _resolve_positions(
            raw.positions,
            ctx.num_prompt_tokens,
            num_generated_tokens=0,
            where="capture.positions",
        )
    else:
        raise CaptureValidationError(
            "capture.positions must be a string or list of ints"
        )

    # 6. Prefix-cache position rejection. Any explicit position below
    # num_computed_tokens was not forwarded through the model (served
    # from the prefix cache), so we can never capture its residual.
    if isinstance(resolved_positions, list):
        bad = [p for p in resolved_positions if p < ctx.num_computed_tokens]
        if bad:
            raise CaptureValidationError(
                f"capture.positions contains indices that were served "
                f"from the prefix cache and will never be forwarded "
                f"through the model: {bad}. The first uncached position "
                f"for this request is {ctx.num_computed_tokens}."
            )

    mix = _validate_mix(raw, resolved_hooks, ctx.num_hidden_layers)
    return CaptureSpec(hooks=resolved_hooks, positions=resolved_positions, mix=mix)


def slug_path_components(raw: FilesystemCaptureRequest) -> tuple[str, str]:
    """Slug ``tag`` and ``request_id`` for filesystem path use.

    Exposed alongside ``validate_filesystem_request`` so the consumer can
    record the slugs at admission time (when the raw ``FilesystemCaptureRequest``
    is still in hand) for later use at submit/finalize time, since the
    framework's :class:`CaptureSpec` only carries ``hooks`` and ``positions``.
    Raises :class:`CaptureValidationError` on invalid slugs.
    """
    tag_slug = _slug(raw.tag, field="capture.tag")
    request_id_slug = _slug(raw.request_id, field="capture.request_id")
    return tag_slug, request_id_slug


__all__: list[str] = ["validate_filesystem_request", "slug_path_components"]


def _validate_mix(
    raw: FilesystemCaptureRequest,
    resolved_hooks: dict[HookName, list[int]],
    num_hidden_layers: int,
) -> LayerMix | None:
    """Validate the optional ``layer_mix`` spec; ``None`` when not requested.

    ``hooks["layer_mix"]`` and ``mix`` must come together; ``layer_mix`` is only
    ever layer 0; members are real hooks at in-range layers, in strictly
    increasing firing order, without repeats; weights are finite and one per
    member; the only norm is ``layer_norm`` (no affine, eps 1e-5).
    """
    import math

    mix = getattr(raw, "mix", None)
    want = "layer_mix" in resolved_hooks
    if mix is None and not want:
        return None
    if mix is None or not want:
        raise CaptureValidationError(
            "capture.mix and capture.hooks['layer_mix'] must be given together"
        )
    if resolved_hooks["layer_mix"] != [0]:
        raise CaptureValidationError("capture.hooks['layer_mix'] must be [0]")
    if not isinstance(mix, dict):
        raise CaptureValidationError("capture.mix must be an object")
    unknown = set(mix) - {"members", "weights", "norm"}
    if unknown:
        raise CaptureValidationError(f"capture.mix has unknown keys {sorted(unknown)}")
    if mix.get("norm", "layer_norm") != "layer_norm":
        raise CaptureValidationError("capture.mix.norm must be 'layer_norm'")
    members_raw = mix.get("members")
    weights_raw = mix.get("weights")
    if not isinstance(members_raw, list) or not members_raw:
        raise CaptureValidationError("capture.mix.members must be a non-empty list")
    if not isinstance(weights_raw, list) or len(weights_raw) != len(members_raw):
        raise CaptureValidationError(
            "capture.mix.weights must be a list with one weight per member"
        )

    def _tap(m: Any) -> tuple[int, str]:
        if (
            not isinstance(m, (list, tuple))
            or len(m) != 2
            or not isinstance(m[0], int)
            or isinstance(m[0], bool)
            or m[1] not in _MIX_MEMBER_HOOKS
        ):
            raise CaptureValidationError(
                f"capture.mix.members taps must be [layer, hook] with hook in "
                f"{sorted(_MIX_MEMBER_HOOKS)}, got {m!r}"
            )
        if not 0 <= m[0] < num_hidden_layers:
            raise CaptureValidationError(
                f"capture.mix.members layer {m[0]} out of range "
                f"[0, {num_hidden_layers})"
            )
        return (int(m[0]), str(m[1]))

    # A member is one tap ``[layer, hook]`` or a group ``[[layer, hook], ...]``
    # summed before normalizing.
    members: list[tuple[tuple[int, str], ...]] = []
    for m in members_raw:
        is_group = (
            isinstance(m, (list, tuple))
            and len(m) > 0
            and all(isinstance(x, (list, tuple)) for x in m)
        )
        members.append(tuple(_tap(x) for x in m) if is_group else (_tap(m),))
    # Firing order within a decoder layer: pre_attn -> post_attn -> mlp_out ->
    # post_block -> post_norm (stack-final, last layer only).
    order = {
        "pre_attn": 0,
        "post_attn": 1,
        "mlp_out": 2,
        "post_block": 3,
        "post_norm": 4,
    }
    ranks = [(layer, order[hook]) for group in members for layer, hook in group]
    if any(b <= a for a, b in zip(ranks, ranks[1:])):
        raise CaptureValidationError(
            "capture.mix.members taps must be distinct and in forward firing order"
        )
    weights: list[float] = []
    for w in weights_raw:
        if (
            isinstance(w, bool)
            or not isinstance(w, (int, float))
            or not math.isfinite(w)
        ):
            raise CaptureValidationError(
                f"capture.mix.weights must be finite numbers, got {w!r}"
            )
        weights.append(float(w))
    return LayerMix(members=tuple(members), weights=tuple(weights))
