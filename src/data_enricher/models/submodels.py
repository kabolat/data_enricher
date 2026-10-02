"""Neural-network building blocks for the experimental VAE."""

from __future__ import annotations

import torch

from ..utils.vae_utils import kl_divergence, log_prob, to_sigma


class NNBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_neurons: int = 50,
        num_hidden_layers: int = 2,
        **_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.num_layers = num_hidden_layers
        self.input_layer = torch.nn.Linear(input_dim, num_neurons)
        self.middle_layers = torch.nn.ModuleList(
            torch.nn.Linear(num_neurons, num_neurons)
            for _ in range(num_hidden_layers)
        )
        self.output_layer = torch.nn.Linear(num_neurons, output_dim)
        self.activation = torch.nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.input_layer(x.reshape(-1, self.input_dim)))
        for layer in self.middle_layers:
            hidden = self.activation(layer(hidden))
        return self.output_layer(hidden)

    def _num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class ParameterizerNN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dist_params: tuple[str, ...] = ("mu",),
        num_hidden_layers: int = 2,
        num_neurons: int = 50,
        **_,
    ):
        super().__init__()
        self.dist_params = dist_params
        # Keep the 0.1.x module names so existing VAE state dictionaries remain loadable.
        self.block_dict = torch.nn.ModuleDict()
        self.block_dict["input"] = NNBlock(
            input_dim, num_neurons, num_neurons, num_hidden_layers
        )
        for parameter in dist_params:
            self.block_dict[parameter] = NNBlock(
                num_neurons, output_dim, num_neurons, 1
            )
        self.activation = torch.nn.ELU()

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.activation(self.block_dict["input"](inputs))
        return {
            parameter: self.block_dict[parameter](hidden)
            for parameter in self.dist_params
        }

    def _num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class GaussianNN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learn_sigma: bool = True,
        num_hidden_layers: int = 2,
        num_neurons: int = 50,
        **_,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        dist_params = ("mu", "sigma") if learn_sigma else ("mu",)
        self.parameterizer = ParameterizerNN(
            input_dim,
            output_dim,
            dist_params=dist_params,
            num_hidden_layers=num_hidden_layers,
            num_neurons=num_neurons,
        )

    def _num_parameters(self) -> int:
        return self.parameterizer._num_parameters()

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        params = self.parameterizer(inputs)
        params["sigma"] = (
            to_sigma(params["sigma"])
            if self.learn_sigma
            else torch.ones_like(params["mu"])
        )
        return params

    def rsample(
        self,
        param_dict: dict[str, torch.Tensor],
        num_samples: int = 1,
        *,
        generator: torch.Generator | None = None,
        **_,
    ) -> torch.Tensor:
        noise = torch.randn(
            (num_samples, *param_dict["mu"].shape),
            device=param_dict["mu"].device,
            dtype=param_dict["mu"].dtype,
            generator=generator,
        )
        return param_dict["mu"] + param_dict["sigma"] * noise

    def sample(self, *args, **kwargs) -> torch.Tensor:
        return self.rsample(*args, **kwargs)

    def log_likelihood(
        self, targets: torch.Tensor, param_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return log_prob("Normal", param_dict, targets)

    def kl_divergence(
        self,
        param_dict: dict[str, torch.Tensor],
        prior_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return kl_divergence("Normal", param_dict, prior_params)


def get_distribution_model(dist_type: str, **kwargs) -> GaussianNN:
    if dist_type.lower() in {"gaussian", "gauss", "normal", "n", "g"}:
        return GaussianNN(**kwargs)
    raise NotImplementedError(f"unknown distribution type: {dist_type}")


__all__ = ["GaussianNN", "NNBlock", "ParameterizerNN", "get_distribution_model"]
