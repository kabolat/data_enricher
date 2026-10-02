import pytest
import torch

from data_enricher import energy_test, mmd_test, model_comparison, sample_comparison


def test_energy_distance_is_zero_for_identical_samples():
    samples = torch.tensor([[0.0], [1.0], [2.0]])
    assert energy_test(samples, samples).item() == pytest.approx(0.0)


def test_comparison_statistics_are_symmetric():
    first = torch.tensor([[0.0], [1.0], [3.0]])
    second = torch.tensor([[1.0], [2.0], [4.0]])
    assert energy_test(first, second) == pytest.approx(energy_test(second, first))
    assert mmd_test(first, second) == pytest.approx(mmd_test(second, first))


def test_sample_comparison_is_reproducible_and_accepts_unequal_pool_sizes():
    train = torch.arange(20, dtype=torch.float32).reshape(-1, 1)
    model = torch.arange(30, dtype=torch.float32).reshape(-1, 1)
    test = torch.arange(10, dtype=torch.float32).reshape(-1, 1)
    first = sample_comparison(
        model, train, test, mc_runs=4, generator=torch.Generator().manual_seed(7)
    )
    second = sample_comparison(
        model, train, test, mc_runs=4, generator=torch.Generator().manual_seed(7)
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_model_comparison_returns_python_float():
    base = torch.tensor([0.0, 1.0, 2.0])
    model = torch.tensor([1.0, 2.0, 3.0])
    assert model_comparison(model, base, test="mean") == pytest.approx(1.0)
    assert isinstance(model_comparison(model, base), float)
