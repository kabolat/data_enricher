import pytest
import torch

from data_enricher import (
    AdaptiveKernelDensityEstimator,
    PiKernelDensityEstimator,
    VanillaKernelDensityEstimator,
)


DATA = torch.tensor([[-1.0], [-0.25], [0.5], [1.5]])


def test_kde_state_and_sampling_are_reproducible():
    model = VanillaKernelDensityEstimator(DATA, sigma=0.3)
    assert set(model.state_dict()) == {"mu", "pitilde", "sigmatilde"}
    first = model.sample(5, generator=torch.Generator().manual_seed(11))
    second = model.sample(5, generator=torch.Generator().manual_seed(11))
    assert torch.equal(first, second)


def test_adaptive_kde_eval_and_em_fit():
    model = AdaptiveKernelDensityEstimator(DATA)
    assert model.eval() is model
    model.fit(DATA, method="em", max_iterations=10)
    assert model.n_iter_ <= 10
    assert torch.isfinite(torch.tensor(model.history_)).all()
    assert torch.all(model.get_params()["sigma"] > 0)


def test_pi_kde_weights_remain_normalized():
    model = PiKernelDensityEstimator(DATA).fit(DATA, max_iterations=5)
    assert model.get_params()["pi"].sum().item() == pytest.approx(1.0)


def test_loo_warns_when_paper_assumption_is_violated():
    repeated = torch.tensor([[0.0], [0.0], [1.0]])
    model = AdaptiveKernelDensityEstimator(repeated)
    with pytest.warns(UserWarning, match="singularity guarantee"):
        model.fit(repeated, max_iterations=1)


def test_imputation_does_not_mutate_input():
    model = VanillaKernelDensityEstimator(DATA, sigma=0.3)
    missing = torch.tensor([[float("nan")], [0.2]])
    result = model.impute(
        missing, num_steps=2, generator=torch.Generator().manual_seed(3)
    )
    assert torch.isnan(missing[0, 0])
    assert torch.isfinite(result).all()


def test_legacy_train_call_remains_available():
    model = AdaptiveKernelDensityEstimator(DATA)
    with pytest.deprecated_call(match="use fit"):
        returned = model.train(
            x=DATA,
            modified_em=True,
            wait_convergence=False,
            num_iterations=2,
            bacth_size=2,
        )
    assert returned is model
    assert model.n_iter_ == 2
