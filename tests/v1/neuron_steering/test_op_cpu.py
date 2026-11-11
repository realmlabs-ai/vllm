"""CPU checks of per-request neuron steering (spec strings over a fixed ranking).

1. The op equals Veer's cpc/steer.py formulas (reflect / amplify), written out per token.
2. The op equals the previous preset op (commit f90705fc9) bit for bit on the same steering.
3. Unsteered tokens and neurons outside a request's top k are bit-identical.
4. Spec parsing: accepted and rejected strings.
"""
import pytest
import torch

import vllm.model_executor.layers.neuron_steering as ns
from vllm.config.neuron_steering import NeuronSteeringConfig, _validate

WIDTH, T = 64, 12
torch.manual_seed(0)
RANKING = {  # 5 ranked neurons over layers 3 and 4, mixed orientations
    "neurons": [[3, 5], [4, 1], [3, 9], [3, 40], [4, 7]],
    "s": [1, -1, -1, 1, 1],
    "qA": [sorted((torch.randn(101) * 0.5).tolist()) for _ in range(5)],
}
n, s, qa = _validate(RANKING)
CFG = NeuronSteeringConfig(neurons=n, s=s, qa=qa)
SPECS = [None, "reflect:top3:0.5:4:all", "amplify:top5:0.9:16:all", "reflect:top1:0.1:0:all",
         "amplify:top2:0.37:2.5:all", "reflect:top5:1:64:all", "amplify:top4:0:8:all"]


def _token_specs():
    return [SPECS[i % len(SPECS)] for i in range(T)]


def _vectors(specs):
    k = torch.zeros(32, dtype=torch.long)
    q = torch.zeros(32, dtype=torch.long)
    c = torch.zeros(32)
    for t, sp in enumerate(specs):
        if sp:
            k[t], q[t], c[t] = CFG.parse(sp)
    return k, q, c


def _op(layer, x, specs):
    tables = ns.build_layer_tables(CFG, layer, WIDTH)
    return ns.apply_neuron_steering(x, *tables, *_vectors(specs))


def _veer(layer, x, specs):
    """Veer's formulas, per token and neuron, straight from the ranking."""
    ref = x.clone()
    for t, sp in enumerate(specs):
        if not sp:
            continue
        fam, k, q, strength, _ = sp.split(":")
        k, q, strength = int(k[3:]), round(float(q) * 100), float(strength)
        for j, (lay, neuron) in enumerate(RANKING["neurons"][:k]):
            if lay != layer:
                continue
            sj = RANKING["s"][j]
            theta = RANKING["qA"][j][q] if sj > 0 else RANKING["qA"][j][100 - q]
            h = x[t, neuron].float()
            if fam == "reflect":   # h - s (1 + alpha) relu(s (h - theta))
                new = h + (-(1.0 + strength)) * sj * torch.relu(sj * (h - theta))
            else:                  # h + s gamma relu(s (h - theta))
                new = h + strength * sj * torch.relu(sj * (h - theta))
            ref[t, neuron] = new.to(x.dtype)
    return ref


@pytest.mark.parametrize("layer", [3, 4])
def test_op_matches_veers_formulas_and_leaves_the_rest_alone(layer):
    x = (torch.randn(T, WIDTH) * 2).to(torch.bfloat16)
    specs = _token_specs()
    out = _op(layer, x, specs)
    assert torch.equal(out, _veer(layer, x, specs))
    unsteered = torch.tensor([sp is None for sp in specs])
    assert torch.equal(out[unsteered], x[unsteered])
    layer_neurons = {nn for lay, nn in RANKING["neurons"] if lay == layer}
    others = [p for p in range(WIDTH) if p not in layer_neurons]
    assert torch.equal(out[:, others], x[:, others])


def _old_preset_op(x, specs, layer):
    """The f90705fc9 op: one preset row per distinct spec, entries [neuron, s, theta, c] padded to K."""
    rows = sorted({sp for sp in specs if sp})
    entries = {}
    for sp in rows:
        k, q, c = CFG.parse(sp)
        entries[sp] = [(nn, CFG.s[j], CFG.qa[j][q] if CFG.s[j] > 0 else CFG.qa[j][100 - q], c)
                       for j, (lay, nn) in enumerate(CFG.neurons[:k]) if lay == layer]
    kmax = max([len(e) for e in entries.values()] + [1])
    R = len(rows) + 1
    idx = torch.zeros(R, kmax, dtype=torch.long)
    sv = torch.ones(R, kmax)
    th = torch.zeros(R, kmax)
    cv = torch.zeros(R, kmax)
    idx[0] = torch.arange(kmax)
    for r, sp in enumerate(rows, start=1):
        used = {e[0] for e in entries[sp]}
        pad = (p for p in range(WIDTH) if p not in used)
        for jj in range(kmax):
            if jj < len(entries[sp]):
                idx[r, jj], sv[r, jj], th[r, jj], cv[r, jj] = entries[sp][jj]
            else:
                idx[r, jj] = next(pad)
    index = torch.zeros(32, dtype=torch.long)
    for t, sp in enumerate(specs):
        index[t] = rows.index(sp) + 1 if sp else 0
    r = index[: x.shape[0]]
    cols = idx[r]
    h = torch.gather(x, 1, cols).float()
    sr = sv[r]
    new = h + cv[r] * sr * torch.relu(sr * (h - th[r]))
    return x.scatter(1, cols, new.to(x.dtype))


@pytest.mark.parametrize("layer", [3, 4])
def test_op_is_bit_identical_to_the_preset_op(layer):
    x = (torch.randn(T, WIDTH) * 2).to(torch.bfloat16)
    specs = _token_specs()
    assert torch.equal(_op(layer, x, specs), _old_preset_op(x, specs, layer))


def test_parse():
    assert CFG.parse("reflect:top3:0.5:4:all") == (3, 50, -5.0)
    assert CFG.parse("amplify:top5:0.37:2.5:all") == (5, 37, 2.5)
    assert CFG.parse("amplify:top1:1:0:all") == (1, 100, 0.0)
    for bad in ["reflect:top0:0.5:4:all", "reflect:top6:0.5:4:all", "reflect:top3:1.5:4:all",
                "reflect:top3:0.5:4:prompt", "push:top3:0.5:4:all", "reflect:top3:0.5:-1:all",
                "reflect:top3:0.5:inf:all", "", "reflect:top3:0.5:4"]:
        with pytest.raises(ValueError):
            CFG.parse(bad)


def test_ranking_validation():
    with pytest.raises(ValueError):
        _validate({**RANKING, "s": [1, 2, 1, 1, 1]})
    with pytest.raises(ValueError):
        _validate({**RANKING, "neurons": RANKING["neurons"][:4]})
    with pytest.raises(ValueError):
        _validate({**RANKING, "qA": [r[:100] for r in RANKING["qA"]]})
    with pytest.raises(ValueError):
        _validate({**RANKING, "neurons": [[3, 5]] * 5})
