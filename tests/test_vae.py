import pytest
import torch

from data_enricher import VAE


def test_vae_eval_fit_and_state_dict():
    model = VAE(input_dim=2, latent_dim=1, num_neurons=4, num_hidden_layers=0)
    data = torch.tensor([[-1.0, 0.0], [0.0, 0.5], [1.0, 1.0]])
    assert model.eval() is model
    assert {"prior_mu", "prior_sigma"} <= set(model.state_dict())
    assert model.fit(data, epochs=1, batch_size=32, verbose_freq=None) is model
    assert len(model.history_) == 1


def test_vae_loads_legacy_state_without_prior_buffers():
    model = VAE(input_dim=2, latent_dim=1, num_neurons=4, num_hidden_layers=0)
    legacy_state = {
        name: value
        for name, value in model.state_dict().items()
        if name not in {"prior_mu", "prior_sigma"}
    }
    restored = VAE(input_dim=2, latent_dim=1, num_neurons=4, num_hidden_layers=0)
    restored.load_state_dict(legacy_state)


def test_vae_sampling_is_reproducible():
    model = VAE(input_dim=2, latent_dim=1, num_neurons=4, num_hidden_layers=0)
    first = model.sample(3, generator=torch.Generator().manual_seed(5))["samples"]
    second = model.sample(3, generator=torch.Generator().manual_seed(5))["samples"]
    assert torch.equal(first, second)


def test_vae_imputation_does_not_mutate_input():
    model = VAE(input_dim=2, latent_dim=1, num_neurons=4, num_hidden_layers=0)
    data = torch.tensor([[float("nan"), 0.0], [1.0, 2.0]])
    result = model.impute(
        data, num_steps=2, generator=torch.Generator().manual_seed(13)
    )
    assert torch.isnan(data[0, 0])
    assert torch.isfinite(result).all()


def test_legacy_vae_train_call_remains_available():
    model = VAE(input_dim=1, latent_dim=1, num_neurons=3, num_hidden_layers=0)
    data = torch.tensor([[-1.0], [0.0], [1.0]])
    with pytest.deprecated_call(match="use fit"):
        returned = model.train(x=data, epochs=1, batch_size=8, verbose_freq=None)
    assert returned is model
