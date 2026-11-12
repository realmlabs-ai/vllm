"""``layer_mix``: a weighted layer-norm sum of member residuals, on the GPU.

The point is to send one float32 ``[rows, hidden]`` tensor out of the GPU
instead of one tensor per tap (MLS's PII head mixes 32 decoder blocks). A
member may be a group of taps summed in the model dtype before normalizing,
which rebuilds a block vLLM never materializes (``post_attn + mlp_out``).
Pinned here:

1. validation accepts exactly ``hooks["layer_mix"] == [0]`` together with a
   well-formed ``mix`` (single taps and groups, in firing order);
2. the emitted tensor is bit-identical to computing the same sum downstream
   from the captured taps (groups added in bf16, then ``w0*ln(r0)``,
   ``acc + w_g*ln(r_g)``, float32, ``F.layer_norm`` without affine);
3. rows of different requests in one step use their own weights; taps also
   requested as outputs are still emitted;
4. a mix whose taps fire out of order or incompletely is never emitted
   (the request gets an error);
5. a mix-only step still forces the eager path, and ``layer_mix`` rows are
   never written to the activation store.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.capture.consumers.filesystem.types import FilesystemCaptureRequest
from vllm.v1.capture.consumers.filesystem.validation import (
    CaptureValidationError,
    _validate_mix,
)
from vllm.v1.capture.manager import _MIX_KEY, CaptureManager
from vllm.v1.capture.plan import CaptureBatchView
from vllm.v1.capture.types import CaptureSpec, LayerMix

NUM_LAYERS = 4
H = 16
# Three single-tap members and one group: block 3 rebuilt as post_attn + mlp_out.
MEMBERS = (
    ((1, "pre_attn"),),
    ((2, "pre_attn"),),
    ((3, "pre_attn"),),
    ((3, "post_attn"), (3, "mlp_out")),
)
TAPS = [t for g in MEMBERS for t in g]
FIRING = ("pre_attn", "post_attn", "mlp_out", "post_block")


def _sink():
    s = MagicMock()
    s.location = "worker"
    s.submit_chunk_batch = None
    s.get_result = MagicMock(return_value=None)
    s.wait_for_result = MagicMock(return_value=None)
    return s


def _manager(dtype=torch.bfloat16):
    sink = _sink()
    mgr = CaptureManager(
        consumers=(sink,),
        consumer_specs=(None,),
        num_hidden_layers=NUM_LAYERS,
        hidden_size=H,
        model_dtype=dtype,
    )
    return mgr, sink


def _spec(weights, positions, extra_hooks=None, members=MEMBERS):
    hooks = {"layer_mix": [0], **(extra_hooks or {})}
    mix = LayerMix(members=members, weights=tuple(weights))
    return CaptureSpec(hooks=hooks, positions=positions, mix=mix)


def _view(req_ids, n_prompt, n_sched):
    offs, run = [], 0
    for n in n_sched:
        offs.append(run)
        run += n
    return CaptureBatchView(
        req_ids=req_ids,
        num_prompt_tokens=n_prompt,
        num_computed_tokens=[0] * len(req_ids),
        num_scheduled_tokens=n_sched,
        token_offsets=offs,
    )


def _hiddens(n_rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        t: (torch.randn(n_rows, H, generator=g) * (3 + i)).to(torch.bfloat16)
        for i, t in enumerate(TAPS)
    }


def _downstream(hs, rows, weights, members=MEMBERS):
    """The sum as a consumer computes it from the captured taps (MLS's pool)."""
    out = None
    for w, group in zip(torch.tensor(weights, dtype=torch.float32), members):
        r = None
        for t in group:
            r = hs[t][rows] if r is None else r + hs[t][rows]  # bf16 add, as HF
        hn = F.layer_norm(r.float(), (H,))
        out = w * hn if out is None else out + w * hn
    return out


def _fire(mgr, hs, skip=()):
    """Fire every hook of a forward, in firing order; taps get their tensors."""
    filler = next(iter(hs.values()))
    for layer in range(NUM_LAYERS):
        for hook in FIRING:
            key = (layer, hook)
            if key in skip:
                continue
            mgr.on_hook(layer, hook, hs.get(key, filler))


def _chunks(sink, key):
    calls = sink.submit_chunk.call_args_list
    return [c.args[0] for c in calls if (c.args[0].key[1], c.args[0].key[2]) == key]


# -- 1. validation --------------------------------------------------


def _raw(mix, hooks=None):
    return FilesystemCaptureRequest(
        request_id="r",
        tag="t",
        hooks=hooks or {"layer_mix": [0]},
        positions="all_prompt",
        mix=mix,
    )


GOOD = {
    "members": [
        [1, "pre_attn"],
        [2, "pre_attn"],
        [3, "pre_attn"],
        [[3, "post_attn"], [3, "mlp_out"]],
    ],
    "weights": [0.1, 0.2, 0.3, 0.4],
    "norm": "layer_norm",
}


def test_validation_accepts_taps_and_groups():
    mix = _validate_mix(_raw(GOOD), {"layer_mix": [0]}, NUM_LAYERS)
    assert mix == LayerMix(members=MEMBERS, weights=(0.1, 0.2, 0.3, 0.4))
    plain = _raw(None, {"pre_attn": [1]})
    assert _validate_mix(plain, {"pre_attn": [1]}, NUM_LAYERS) is None


def _m(*members):
    return {**GOOD, "members": list(members), "weights": [0.25] * len(members)}


@pytest.mark.parametrize(
    "mix,hooks",
    [
        (GOOD, {"pre_attn": [1]}),  # mix without the layer_mix hook
        (None, {"layer_mix": [0]}),  # layer_mix hook without mix
        (GOOD, {"layer_mix": [1]}),  # not at layer 0
        ({**GOOD, "norm": "rms"}, None),
        ({**GOOD, "weights": [0.1, 0.2]}, None),
        ({**GOOD, "weights": [0.1, 0.2, float("nan"), 0.4]}, None),
        (_m([1, "pre_attn"], [1, "layer_mix"]), None),  # not a tap
        (_m([2, "pre_attn"], [1, "pre_attn"]), None),  # order
        (_m([1, "pre_attn"], [1, "pre_attn"]), None),  # repeat
        (_m([1, "pre_attn"], [9, "pre_attn"]), None),  # range
        (_m([[3, "mlp_out"], [3, "post_attn"]]), None),  # group out of order
        (_m([1, "pre_attn"], []), None),  # empty member
        ({**GOOD, "extra": 1}, None),
    ],
)
def test_validation_rejects(mix, hooks):
    hooks = hooks or {"layer_mix": [0]}
    resolved = {k: list(v) for k, v in hooks.items()}
    with pytest.raises(CaptureValidationError):
        _validate_mix(_raw(mix, hooks), resolved, NUM_LAYERS)


# -- 2-5. the manager -----------------------------------------------


def test_mix_is_bit_identical_to_the_downstream_sum():
    mgr, sink = _manager()
    w = [0.05, 0.15, 0.3, 0.5]
    mgr.register_request(
        "r1", client_specs={0: _spec(w, "all_prompt")}, num_prompt_tokens=7
    )
    plan = mgr.build_step_plan(_view(["r1"], [7], [7]))
    assert mgr.has_pending_capture()  # a mix-only step must run eager
    assert _MIX_KEY not in plan.gather_indices  # no forward hook fires for it
    hs = _hiddens(7)
    _fire(mgr, hs)
    mgr.dispatch_step_captures(plan)
    mgr._drain_dispatch_queue()
    (chunk,) = _chunks(sink, _MIX_KEY)
    assert chunk.tensor.dtype == torch.float32
    assert torch.equal(chunk.tensor, _downstream(hs, list(range(7)), w))
    assert not _chunks(sink, (1, "pre_attn"))  # taps are not emitted unasked


def test_group_is_summed_in_model_dtype_before_the_norm():
    """Normalizing the bf16 sum differs from summing normalized parts."""
    hs = _hiddens(5, seed=7)
    rows = list(range(5))
    w = [0.25] * 4
    split = MEMBERS[:3] + (((3, "post_attn"),), ((3, "mlp_out"),))
    assert not torch.equal(
        _downstream(hs, rows, w), _downstream(hs, rows, w + [0.25], split)
    )


def test_two_requests_one_step_use_their_own_weights_and_taps_still_emit():
    mgr, sink = _manager()
    w1, w2 = [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]
    spec_a = _spec(w1, "all_prompt", {"pre_attn": [2]})
    mgr.register_request("a", client_specs={0: spec_a}, num_prompt_tokens=3)
    mgr.register_request(
        "b", client_specs={0: _spec(w2, "all_prompt")}, num_prompt_tokens=4
    )
    plan = mgr.build_step_plan(_view(["a", "b"], [3, 4], [3, 4]))
    hs = _hiddens(7, seed=1)
    _fire(mgr, hs)
    mgr.dispatch_step_captures(plan)
    mgr._drain_dispatch_queue()
    by_req = {c.key[0]: c.tensor for c in _chunks(sink, _MIX_KEY)}
    assert torch.equal(by_req["a"], _downstream(hs, [0, 1, 2], w1))
    assert torch.equal(by_req["b"], _downstream(hs, [3, 4, 5, 6], w2))
    (tap,) = _chunks(sink, (2, "pre_attn"))  # "a" also asked for pre_attn@2
    assert tap.key[0] == "a"
    assert torch.equal(tap.tensor, hs[(2, "pre_attn")][:3])


@pytest.mark.parametrize("case", ["missing_group_tap", "missing_tap", "reorder"])
def test_out_of_order_or_incomplete_mix_is_never_emitted(case):
    mgr, sink = _manager()
    spec = _spec([0.25] * 4, "all_prompt")
    mgr.register_request("r", client_specs={0: spec}, num_prompt_tokens=5)
    plan = mgr.build_step_plan(_view(["r"], [5], [5]))
    hs = _hiddens(5, seed=2)
    if case == "missing_group_tap":  # mlp_out never fires (e.g. a dropped op)
        _fire(mgr, hs, skip={(3, "mlp_out")})
    elif case == "missing_tap":
        _fire(mgr, hs, skip={(2, "pre_attn")})
    else:
        mgr.on_hook(2, "pre_attn", hs[(2, "pre_attn")])  # #1 before #0
        _fire(mgr, hs)
    mgr.dispatch_step_captures(plan)
    mgr._drain_dispatch_queue()
    assert not _chunks(sink, _MIX_KEY)
    err = mgr._requests["r"].error
    assert err and "layer_mix" in err


def test_layer_mix_rows_are_never_written_to_the_store(monkeypatch):
    from vllm.v1.capture import manager as M

    store = MagicMock()
    monkeypatch.setattr(M, "get_active_activation_store", lambda: store)
    monkeypatch.setattr(M, "activation_key", lambda *a: ("k",) + a[2:])
    mgr, _ = _manager()
    spec = _spec([0.25] * 4, "all_prompt", {"pre_attn": [1]})
    mgr.register_request(
        "r",
        client_specs={0: spec},
        num_prompt_tokens=3,
        block_hashes=[b"x"],
        hash_block_size=16,
    )
    plan = mgr.build_step_plan(_view(["r"], [3], [3]))
    _fire(mgr, _hiddens(3, seed=3))
    mgr.dispatch_step_captures(plan)
    mgr._drain_dispatch_queue()
    hooks = {c.args[0][-1] for c in store.put.call_args_list}
    assert "pre_attn" in hooks and "layer_mix" not in hooks


def test_mixed_members_across_requests_fail_rather_than_cross_mix():
    mgr, sink = _manager()
    other = _spec([0.3, 0.3, 0.4], "all_prompt", members=MEMBERS[:3])
    mgr.register_request(
        "a", client_specs={0: _spec([0.25] * 4, "all_prompt")}, num_prompt_tokens=2
    )
    mgr.register_request("b", client_specs={0: other}, num_prompt_tokens=2)
    plan = mgr.build_step_plan(_view(["a", "b"], [2, 2], [2, 2]))
    assert plan.mix is None
    _fire(mgr, _hiddens(4, seed=4))
    mgr.dispatch_step_captures(plan)
    mgr._drain_dispatch_queue()
    assert not _chunks(sink, _MIX_KEY)
    assert mgr._requests["a"].error and mgr._requests["b"].error


def test_spec_filter_keeps_mix_on_a_full_stage():
    from vllm.v1.capture.manager import _filter_specs_to_layer_range

    spec = _spec([0.25] * 4, "all_prompt")
    kept = _filter_specs_to_layer_range({0: spec}, 0, NUM_LAYERS)
    assert kept[0].mix == spec.mix
    # Members split across pipeline stages: the mix is refused.
    assert _filter_specs_to_layer_range({0: spec}, 0, 2) == {}
