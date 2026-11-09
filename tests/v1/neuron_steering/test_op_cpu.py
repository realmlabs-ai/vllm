"""CPU check of torch.ops.vllm.apply_neuron_steering against Veer's cpc/steer.py formulas (reflect/amplify)."""
import json, torch
from vllm.config.neuron_steering import NeuronSteeringConfig, _validate
import vllm.model_executor.layers.neuron_steering as ns

torch.manual_seed(0)
I, T = 64, 10
presets = {"away": {"layers": {"3": [[5, 1, 0.2, -3.0], [9, -1, -0.1, -3.0], [40, 1, 0.0, -3.0]]}},
           "toward": {"layers": {"3": [[5, 1, 0.2, 2.0]], "4": [[1, -1, 0.3, 2.0]]}}}
cfg = NeuronSteeringConfig(presets=_validate(presets))
idx, s, th, c = ns.build_layer_tables(cfg, 3, I)
x = (torch.randn(T, I) * 2).to(torch.bfloat16)
rows = torch.tensor([0, 1, 2, 1, 0, 2, 2, 1, 0, 0])            # mix of none / away / toward per token
index = torch.zeros(32, dtype=torch.long); index[:T] = rows
out = ns.apply_neuron_steering(x, idx, s, th, c, index)  # the registered op's implementation (CUDA-only dispatch key)

ref = x.clone()
for t in range(T):
    name = cfg.names[rows[t] - 1] if rows[t] else None
    for n, sj, tj, cj in (cfg.presets[name].get(3, []) if name else []):
        h = x[t, n].float()
        if cj < 0:   # reflect: h - s (1+alpha) relu(s (h - theta)), c = -(1+alpha)
            new = h - sj * (-cj) * torch.relu(sj * (h - tj))
        else:        # amplify: h + s gamma relu(s (h - theta)), c = gamma
            new = h + sj * cj * torch.relu(sj * (h - tj))
        ref[t, n] = new.to(torch.bfloat16)
assert torch.equal(out, ref), (out - ref).abs().max()
assert torch.equal(out[rows == 0], x[rows == 0])                 # unsteered tokens bit-identical
changed = (out != x)
assert changed[:, [p for p in range(I) if p not in (5, 9, 40)]].sum() == 0  # nothing outside the preset's neurons
print("op == reference; unsteered rows identical; padding untouched; steered elements changed:", int(changed.sum()))
print("names/rows:", cfg.names, [cfg.row_of(n) for n in cfg.names], "layers", cfg.layers, "K", cfg.max_neurons_per_layer)
