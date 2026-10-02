"""Statistical comparisons used to evaluate generated samples."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from scipy.stats import cramervonmises_2samp, ks_2samp


def _validate_samples(*samples: torch.Tensor) -> None:
    for sample in samples:
        if not isinstance(sample, torch.Tensor) or sample.ndim != 2:
            raise ValueError("samples must be two-dimensional torch tensors")
        if sample.shape[0] < 2:
            raise ValueError("each sample set must contain at least two rows")
        if not sample.is_floating_point():
            raise ValueError("samples must use a floating-point dtype")
        if not torch.isfinite(sample).all():
            raise ValueError("samples must contain only finite values")

    reference = samples[0]
    for sample in samples[1:]:
        if sample.shape[1] != reference.shape[1]:
            raise ValueError("sample sets must have the same number of features")
        if sample.device != reference.device or sample.dtype != reference.dtype:
            raise ValueError("sample sets must have the same device and dtype")


def mmd_test(
    base_samples: torch.Tensor,
    test_samples: torch.Tensor,
    alphas: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0),
) -> torch.Tensor:
    """Return the unbiased squared maximum mean discrepancy statistic."""
    _validate_samples(base_samples, test_samples)
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise ValueError("alphas must contain positive values")

    n_base, n_test = base_samples.shape[0], test_samples.shape[0]
    samples = torch.cat((base_samples, test_samples))
    squared_distances = torch.cdist(samples, samples).square()
    kernels = sum(torch.exp(-alpha * squared_distances) for alpha in alphas)

    base_kernel = kernels[:n_base, :n_base]
    test_kernel = kernels[n_base:, n_base:]
    cross_kernel = kernels[:n_base, n_base:]
    return (
        (base_kernel.sum() - base_kernel.diagonal().sum()) / (n_base * (n_base - 1))
        + (test_kernel.sum() - test_kernel.diagonal().sum())
        / (n_test * (n_test - 1))
        - 2 * cross_kernel.mean()
    )


def energy_test(
    base_samples: torch.Tensor, test_samples: torch.Tensor
) -> torch.Tensor:
    """Return the multivariate energy-distance statistic."""
    _validate_samples(base_samples, test_samples)

    n_base = base_samples.shape[0]
    samples = torch.cat((base_samples, test_samples))
    distances = torch.cdist(samples, samples)
    base_distances = distances[:n_base, :n_base]
    test_distances = distances[n_base:, n_base:]
    cross_distances = distances[:n_base, n_base:]
    return 2 * cross_distances.mean() - base_distances.mean() - test_distances.mean()


def sample_comparison(
    model_samples: torch.Tensor,
    train_samples: torch.Tensor,
    test_samples: torch.Tensor,
    test: str = "mmd",
    subsample_ratio: float = 0.4,
    mc_runs: int = 1000,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare random subsets of generated and training data with test data."""
    _validate_samples(model_samples, train_samples, test_samples)
    if not 0 < subsample_ratio <= 1:
        raise ValueError("subsample_ratio must be in (0, 1]")
    if mc_runs < 1:
        raise ValueError("mc_runs must be positive")

    test_functions = {"mmd": mmd_test, "energy": energy_test}
    try:
        test_function = test_functions[test.lower()]
    except KeyError as error:
        raise ValueError(f"unknown sample comparison test: {test}") from error

    subsample_size = max(2, int(test_samples.shape[0] * subsample_ratio))
    if any(sample.shape[0] < subsample_size for sample in (model_samples, train_samples)):
        raise ValueError("model and training samples must cover the subsample size")

    base_scores = test_samples.new_empty(mc_runs)
    model_scores = test_samples.new_empty(mc_runs)
    for index in range(mc_runs):
        test_subset = test_samples[
            torch.randperm(test_samples.shape[0], device=test_samples.device, generator=generator)[
                :subsample_size
            ]
        ]
        train_subset = train_samples[
            torch.randperm(
                train_samples.shape[0], device=train_samples.device, generator=generator
            )[:subsample_size]
        ]
        model_subset = model_samples[
            torch.randperm(
                model_samples.shape[0], device=model_samples.device, generator=generator
            )[:subsample_size]
        ]
        base_scores[index] = test_function(test_subset, train_subset)
        model_scores[index] = test_function(test_subset, model_subset)
    return model_scores, base_scores


def model_comparison(
    model_scores: torch.Tensor, base_scores: torch.Tensor, test: str = "ks"
) -> float:
    """Compare the distributions of model and baseline test statistics."""
    if model_scores.ndim != 1 or base_scores.ndim != 1:
        raise ValueError("model_scores and base_scores must be one-dimensional")
    model = model_scores.detach().cpu().numpy()
    base = base_scores.detach().cpu().numpy()

    test = test.lower()
    if test == "ks":
        return float(ks_2samp(base, model).statistic)
    if test == "cvm":
        return float(cramervonmises_2samp(base, model).statistic)
    if test == "mean":
        return float(model.mean() - base.mean())
    raise ValueError(f"unknown model comparison test: {test}")


# Backward-compatible 0.1.x names. Remove in 1.0.
MMDTest = mmd_test
EnergyTest = energy_test

__all__ = [
    "EnergyTest",
    "MMDTest",
    "energy_test",
    "mmd_test",
    "model_comparison",
    "sample_comparison",
]
