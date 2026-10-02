"""Deterministic smoke check for the LOO-MLL stability intuition."""

import torch

from data_enricher import AdaptiveKernelDensityEstimator


data = torch.tensor([[-1.0], [-0.2], [0.6], [1.4]])
bandwidths = torch.logspace(0, -4, steps=20)
regular_objectives = []
loo_objectives = []

for bandwidth in bandwidths:
    model = AdaptiveKernelDensityEstimator(data, sigma=bandwidth)
    regular_objectives.append(float(model(data, leave_one_out=False).mean()))
    loo_objectives.append(float(model(data, leave_one_out=True).mean()))

assert regular_objectives[-1] > regular_objectives[0]
assert loo_objectives[-1] < loo_objectives[0]

print("As bandwidth approaches zero:")
print(f"  regular MLL: {regular_objectives[0]:.3f} -> {regular_objectives[-1]:.3f}")
print(f"  LOO-MLL:     {loo_objectives[0]:.3f} -> {loo_objectives[-1]:.3f}")
