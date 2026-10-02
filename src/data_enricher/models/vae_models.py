"""Experimental variational autoencoder models.

These models are useful package extras but are not part of the KDE method
validated by Bölat et al. (2024).
"""

from __future__ import annotations

import warnings

import torch

from .submodels import get_distribution_model


class VAE(torch.nn.Module):
    """Diagonal-Gaussian VAE with optional conditioning."""

    def __init__(
        self,
        input_dim: int,
        cond_dim: int = 0,
        latent_dim: int = 3,
        num_neurons: int = 20,
        num_hidden_layers: int = 2,
        learn_decoder_sigma: bool = True,
    ):
        super().__init__()
        if input_dim < 1 or cond_dim < 0 or latent_dim < 1:
            raise ValueError("input_dim and latent_dim must be positive; cond_dim cannot be negative")

        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.num_neurons = num_neurons
        self.num_hidden_layers = num_hidden_layers
        self.learn_decoder_sigma = learn_decoder_sigma
        network_options = {
            "num_neurons": num_neurons,
            "num_hidden_layers": num_hidden_layers,
        }
        self.encoder = get_distribution_model(
            "normal",
            input_dim=input_dim + cond_dim,
            output_dim=latent_dim,
            learn_sigma=True,
            **network_options,
        )
        self.decoder = get_distribution_model(
            "normal",
            input_dim=latent_dim + cond_dim,
            output_dim=input_dim,
            learn_sigma=learn_decoder_sigma,
            **network_options,
        )
        self.register_buffer("prior_mu", torch.zeros(latent_dim))
        self.register_buffer("prior_sigma", torch.ones(latent_dim))
        self.num_parameters = sum(parameter.numel() for parameter in self.parameters())
        self.history_: list[dict[str, float]] = []

    @property
    def prior_params(self) -> dict[str, torch.Tensor]:
        return {"mu": self.prior_mu, "sigma": self.prior_sigma}

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # 0.1.x did not persist the fixed standard-normal prior.
        state_dict.setdefault(prefix + "prior_mu", self.prior_mu)
        state_dict.setdefault(prefix + "prior_sigma", self.prior_sigma)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _conditions(
        self, inputs: torch.Tensor, conditions: torch.Tensor | None
    ) -> torch.Tensor:
        if self.cond_dim == 0:
            return inputs.new_empty((inputs.shape[0], 0))
        if conditions is None:
            raise ValueError("conditions must be provided for a conditional VAE")
        if conditions.shape != (inputs.shape[0], self.cond_dim):
            raise ValueError("conditions have the wrong shape")
        return conditions.to(device=inputs.device, dtype=inputs.dtype)

    def forward(
        self,
        inputs: torch.Tensor,
        conditions: torch.Tensor | None = None,
        mc_samples: int = 1,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[dict, dict]:
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError("inputs have the wrong shape")
        if mc_samples < 1:
            raise ValueError("mc_samples must be positive")
        conditions = self._conditions(inputs, conditions)
        posterior = self.encoder(torch.cat((inputs, conditions), dim=1))
        latent = self.encoder.rsample(
            posterior, num_samples=mc_samples, generator=generator
        )
        repeated_conditions = conditions.unsqueeze(0).expand(mc_samples, -1, -1)
        likelihood = self.decoder(torch.cat((latent, repeated_conditions), dim=2))
        likelihood = {
            name: value.reshape(mc_samples, inputs.shape[0], self.input_dim)
            for name, value in likelihood.items()
        }
        return {"params": likelihood}, {"params": posterior, "samples": latent}

    def sample(
        self,
        num_samples_prior: int = 1,
        num_samples_likelihood: int = 1,
        conditions: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, dict | torch.Tensor]:
        if num_samples_prior < 1 or num_samples_likelihood < 1:
            raise ValueError("sample counts must be positive")
        if self.cond_dim == 0:
            condition = self.prior_mu.new_empty((num_samples_prior, 1, 0))
        else:
            if conditions is None or conditions.shape != (self.cond_dim,):
                raise ValueError("conditions must be a one-dimensional conditional vector")
            condition = conditions.to(self.prior_mu).reshape(1, 1, -1).expand(
                num_samples_prior, -1, -1
            )

        with torch.no_grad():
            latent = self.encoder.sample(
                self.prior_params,
                num_samples=num_samples_prior,
                generator=generator,
            ).unsqueeze(1)
            params = self.decoder(torch.cat((latent, condition), dim=2))
            samples = self.decoder.sample(
                params,
                num_samples=num_samples_likelihood,
                generator=generator,
            )
        return {"params": params, "samples": samples}

    def impute(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor | None = None,
        num_steps: int = 100,
        use_mean: bool = False,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        result = x.clone()
        missing = result.isnan()
        if not missing.any():
            return result
        result[missing] = 0
        for _ in range(num_steps):
            reconstruction, _ = self.reconstruct(
                result, conditions=conditions, generator=generator
            )
            values = (
                reconstruction["params"]["mu"].squeeze(0)
                if use_mean
                else reconstruction["samples"].squeeze(0).squeeze(0)
            )
            result[missing] = values[missing]
        return result

    def reconstruct(
        self,
        inputs: torch.Tensor,
        conditions: torch.Tensor | None = None,
        mc_samples: int = 1,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[dict, dict]:
        with torch.no_grad():
            reconstruction, latent = self.forward(
                inputs, conditions, mc_samples, generator=generator
            )
            reconstruction["samples"] = self.decoder.sample(
                reconstruction["params"], num_samples=1, generator=generator
            )
        return reconstruction, latent

    def reconstruction_loglikelihood(
        self, x: torch.Tensor, likelihood_params: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.decoder.log_likelihood(x, likelihood_params).sum(dim=2).mean(dim=0)

    def kl_divergence(
        self,
        posterior_params: dict[str, torch.Tensor],
        prior_params: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        prior = self.prior_params if prior_params is None else prior_params
        return self.encoder.kl_divergence(posterior_params, prior).sum(dim=1)

    def loss(
        self,
        x: torch.Tensor,
        likelihood_params: dict[str, torch.Tensor],
        posterior_params: dict[str, torch.Tensor],
        prior_params: dict[str, torch.Tensor] | None = None,
        beta: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        reconstruction = self.reconstruction_loglikelihood(x, likelihood_params).mean()
        divergence = self.kl_divergence(posterior_params, prior_params).mean()
        return {
            "loss": -(reconstruction - beta * divergence),
            "elbo": reconstruction - divergence,
            "rll": reconstruction,
            "kl": divergence,
        }

    def fit(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor | None = None,
        *,
        beta: float = 1.0,
        mc_samples: int = 1,
        learning_rate: float = 1e-3,
        epochs: int = 500,
        verbose_freq: int | None = 100,
        batch_size: int = 32,
        generator: torch.Generator | None = None,
    ) -> "VAE":
        """Fit the experimental VAE and return the model."""
        if x.ndim != 2 or x.shape[1] != self.input_dim or not torch.isfinite(x).all():
            raise ValueError("x must be a finite two-dimensional tensor of input_dim features")
        conditions = self._conditions(x, conditions)
        if epochs < 1 or batch_size < 1 or learning_rate <= 0 or mc_samples < 1:
            raise ValueError("training arguments must be positive")

        super().train(True)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, conditions),
            batch_size=min(batch_size, len(x)),
            shuffle=True,
            generator=generator,
        )
        self.history_ = []
        total_iterations = len(dataloader) * epochs
        iteration = 0
        for _ in range(epochs):
            for inputs, batch_conditions in dataloader:
                iteration += 1
                reconstruction, latent = self.forward(
                    inputs,
                    batch_conditions,
                    mc_samples=mc_samples,
                    generator=generator,
                )
                optimizer.zero_grad()
                losses = self.loss(
                    inputs,
                    reconstruction["params"],
                    latent["params"],
                    beta=beta,
                )
                losses["loss"].backward()
                optimizer.step()
                record = {name: float(value.detach()) for name, value in losses.items()}
                self.history_.append(record)
                if verbose_freq and iteration % verbose_freq == 0:
                    print(
                        f"Iteration: {iteration}/{total_iterations} -- "
                        f"ELBO={record['elbo']:.2e} / RLL={record['rll']:.2e} / "
                        f"KL={record['kl']:.2e}"
                    )
        return self

    def train(self, mode: bool | torch.Tensor = True, *args, **kwargs):
        """Set module mode, or temporarily support the deprecated 0.1 fit API."""
        if isinstance(mode, bool) and not args and "x" not in kwargs:
            return super().train(mode)
        warnings.warn(
            "train(x=...) is deprecated; use fit(x, ...) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        x = kwargs.pop("x", mode if isinstance(mode, torch.Tensor) else None)
        if x is None or args:
            raise TypeError("legacy training requires x as the only positional argument")
        return self.fit(x, **kwargs)


__all__ = ["VAE"]
