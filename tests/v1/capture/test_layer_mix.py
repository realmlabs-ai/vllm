"""``layer_mix``: a weighted layer-norm sum of member residuals, accumulated on the GPU in ``on_hook``.

The point is to send one float32 ``[rows, hidden]`` tensor out of the GPU instead of one tensor per member (MLS's PII
head mixes 32 layers). The contract these tests pin:

1. validation accepts exactly ``hooks["layer_mix"] == [0]`` together with a well-formed ``mix``;
2. the emitted tensor is bit-identical to computing the same sum downstream from the captured members
   (``w0*ln(h0)``, then ``acc + w_i*ln(h_i)``, float32, ``F.layer_norm`` without affine);
3. rows of different requests in one step use their own weights; members also requested as outputs are still emitted;
4. a mix whose members fire out of order or incompletely is never emitted (the request gets an error);
5. a mix-only step still forces the eager path, and ``layer_mix`` rows are never written to the activation store.
"""
from __future__ import annotations

from types import SimpleNamespace
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
MEMBERS = ((1, "pre_attn"), (2, "pre_attn"), (3, "pre_attn"), (3, "post_block"))


def _sink():
    s = MagicMock()
    s.location = "worker"
    s.submit_chunk_batch = None
    s.get_result = MagicMock(return_value=None)
    s.wait_for_result = MagicMock(return_value=None)
    return s


def _manager(dtype=torch.bfloat16):
    sink = _sink()
    mgr = CaptureManager(consumers=(sink,), consumer_specs=(None,), num_hidden_layers=NUM_LAYERS,
                         hidden_size=H, model_dtype=dtype)
    return mgr, sink


def _spec(weights, positions, extra_hooks=None):
    hooks = {"layer_mix": [0], **(extra_hooks or {})}
    return CaptureSpec(hooks=hooks, positions=positions, mix=LayerMix(members=MEMBERS, weights=tuple(weights)))


def _view(req_ids, n_prompt, n_sched):
    offs, run = [], 0
    for n in n_sched:
        offs.append(run)
        run += n
    return CaptureBatchView(req_ids=req_ids, num_prompt_tokens=n_prompt, num_computed_tokens=[0] * len(req_ids),
                            num_scheduled_tokens=n_sched, token_offsets=offs)


def _hiddens(n_rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {m: (torch.randn(n_rows, H, generator=g) * (3 + i)).to(torch.bfloat16) for i, m in enumerate(MEMBERS)}


def _downstream(hs, rows, weights):
    """The sum as a consumer computes it from the captured members (MLS's ElmoPool)."""
    out = None
    for w, m in zip(torch.tensor(weights, dtype=torch.float32), MEMBERS):
        hn = F.layer_norm(hs[m][rows].float(), (H,))
        out = w * hn if out is None else out + w * hn
    return out


def _fire(mgr, hs, order=MEMBERS):
    # Interleave with the non-member hooks a real forward fires, in firing order.
    for layer in range(NUM_LAYERS):
        for hook in ("pre_attn", "post_attn", "post_block"):
            key = (layer, hook)
            h = hs.get(key)
            if key in order and h is not None:
                mgr.on_hook(layer, hook, h)
            elif h is None:
                mgr.on_hook(layer, hook, next(iter(hs.values())))


def _chunks(sink, key):
    return [c.args[0] for c in sink.submit_chunk.call_args_list if (c.args[0].key[1], c.args[0].key[2]) == key]


# -- 1. validation ---------------------------------------------------------------------------------------------

def _raw(mix, hooks=None):
    return FilesystemCaptureRequest(request_id="r", tag="t", hooks=hooks or {"layer_mix": [0]},
                                    positions="all_prompt", mix=mix)


GOOD = {"members": [list(m) for m in MEMBERS], "weights": [0.1, 0.2, 0.3, 0.4], "norm": "layer_norm"}


def test_validation_accepts_a_well_formed_mix():
    mix = _validate_mix(_raw(GOOD), {"layer_mix": [0]}, NUM_LAYERS)
    assert mix == LayerMix(members=MEMBERS, weights=(0.1, 0.2, 0.3, 0.4))
    assert _validate_mix(_raw(None, {"pre_attn": [1]}), {"pre_attn": [1]}, NUM_LAYERS) is None


@pytest.mark.parametrize("mix,hooks,why", [
    (GOOD, {"pre_attn": [1]}, "mix without the layer_mix hook"),
    (None, {"layer_mix": [0]}, "layer_mix hook without mix"),
    (GOOD, {"layer_mix": [1]}, "layer_mix not at layer 0"),
    ({**GOOD, "norm": "rms"}, None, "unknown norm"),
    ({**GOOD, "weights": [0.1, 0.2]}, None, "weights length"),
    ({**GOOD, "weights": [0.1, 0.2, float("nan"), 0.4]}, None, "non-finite weight"),
    ({**GOOD, "members": [[1, "pre_attn"], [1, "layer_mix"], [2, "pre_attn"], [3, "pre_attn"]]}, None, "member hook"),
    ({**GOOD, "members": [[2, "pre_attn"], [1, "pre_attn"], [3, "pre_attn"], [3, "post_block"]]}, None, "order"),
    ({**GOOD, "members": [[1, "pre_attn"], [1, "pre_attn"], [3, "pre_attn"], [3, "post_block"]]}, None, "repeat"),
    ({**GOOD, "members": [[1, "pre_attn"], [2, "pre_attn"], [3, "pre_attn"], [9, "post_block"]]}, None, "range"),
    ({**GOOD, "extra": 1}, None, "unknown key"),
])
def test_validation_rejects(mix, hooks, why):
    hooks = hooks or {"layer_mix": [0]}
    resolved = {k: list(v) for k, v in hooks.items()}
    with pytest.raises(CaptureValidationError):
        _validate_mix(_raw(mix, hooks), resolved, NUM_LAYERS)


# -- 2-5. the manager --------------------------------------------------------------------------------------------

def test_mix_is_bit_identical_to_the_downstream_sum():
    mgr, sink = _manager()
    w = [0.05, 0.15, 0.3, 0.5]
    mgr.register_request("r1", client_specs={0: _spec(w, "all_prompt")}, num_prompt_tokens=7)
    plan = mgr.build_step_plan(_view(["r1"], [7], [7]))
    assert mgr.has_pending_capture()                     # a mix-only step must run eager
    assert _MIX_KEY not in plan.gather_indices            # no forward hook fires for it
    hs = _hiddens(7)
    _fire(mgr, hs)
    mgr.dispatch_step_captures(plan)
    mgr._drain_dispatch_queue()
    (chunk,) = _chunks(sink, _MIX_KEY)
    assert chunk.tensor.dtype == torch.float32
    assert torch.equal(chunk.tensor, _downstream(hs, list(range(7)), w))
    assert not _chunks(sink, (1, "pre_attn"))            # members are not emitted unless asked for


def test_two_requests_one_step_use_their_own_weights_and_members_still_emit():
    mgr, sink = _manager()
    w1, w2 = [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]
    mgr.register_request("a", client_specs={0: _spec(w1, "all_prompt", {"pre_attn": [2]})}, num_prompt_tokens=3)
    mgr.register_request("b", client_specs={0: _spec(w2, "all_prompt")}, num_prompt_tokens=4)
    plan = mgr.build_step_plan(_view(["a", "b"], [3, 4], [3, 4]))
    hs = _hiddens(7, seed=1)
    _fire(mgr, hs)
    mgr.dispatch_step_captures(plan)
    mgr._drain_dispatch_queue()
    by_req = {c.key[0]: c.tensor for c in _chunks(sink, _MIX_KEY)}
    assert torch.equal(by_req["a"], _downstream(hs, [0, 1, 2], w1))
    assert torch.equal(by_req["b"], _downstream(hs, [3, 4, 5, 6], w2))
    (member,) = _chunks(sink, (2, "pre_attn"))            # "a" also asked for pre_attn@2 itself
    assert member.key[0] == "a" and torch.equal(member.tensor, hs[(2, "pre_attn")][:3])


def test_out_of_order_or_incomplete_mix_is_never_emitted():
    for order in (MEMBERS[:3], MEMBERS):                  # incomplete; then out of order (below)
        mgr, sink = _manager()
        mgr.register_request("r", client_specs={0: _spec([0.25] * 4, "all_prompt")}, num_prompt_tokens=5)
        plan = mgr.build_step_plan(_view(["r"], [5], [5]))
        hs = _hiddens(5, seed=2)
        if order is MEMBERS:
            mgr.on_hook(2, "pre_attn", hs[(2, "pre_attn")])    # #1 before #0
            for m in MEMBERS:
                mgr.on_hook(m[0], m[1], hs[m])
        else:
            for m in order:
                mgr.on_hook(m[0], m[1], hs[m])
        mgr.dispatch_step_captures(plan)
        mgr._drain_dispatch_queue()
        assert not _chunks(sink, _MIX_KEY)
        assert mgr._requests["r"].error and "layer_mix" in mgr._requests["r"].error


def test_layer_mix_rows_are_never_written_to_the_store(monkeypatch):
    from vllm.v1.capture import manager as M
    store = MagicMock()
    monkeypatch.setattr(M, "get_active_activation_store", lambda: store)
    monkeypatch.setattr(M, "activation_key", lambda *a: ("k",) + a[2:])
    mgr, _ = _manager()
    mgr.register_request("r", client_specs={0: _spec([0.25] * 4, "all_prompt", {"pre_attn": [1]})},
                         num_prompt_tokens=3, block_hashes=[b"x"], hash_block_size=16)
    plan = mgr.build_step_plan(_view(["r"], [3], [3]))
    _fire(mgr, _hiddens(3, seed=3))
    mgr.dispatch_step_captures(plan)
    mgr._drain_dispatch_queue()
    hooks = {c.args[0][-1] for c in store.put.call_args_list}
    assert "pre_attn" in hooks and "layer_mix" not in hooks


def test_mixed_members_across_requests_fail_rather_than_cross_mix():
    mgr, sink = _manager()
    other = CaptureSpec(hooks={"layer_mix": [0]}, positions="all_prompt",
                        mix=LayerMix(members=MEMBERS[:3], weights=(0.3, 0.3, 0.4)))
    mgr.register_request("a", client_specs={0: _spec([0.25] * 4, "all_prompt")}, num_prompt_tokens=2)
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
    assert _filter_specs_to_layer_range({0: spec}, 0, NUM_LAYERS)[0].mix == spec.mix
    assert _filter_specs_to_layer_range({0: spec}, 0, 2) == {}       # members split across stages: refused
