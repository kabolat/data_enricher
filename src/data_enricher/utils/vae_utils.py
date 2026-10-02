"""Distribution helpers for the experimental VAE."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def to_sigma(sigmatilde: torch.Tensor) -> torch.Tensor:
    return F.softplus(sigmatilde)


def from_sigma(sigma: torch.Tensor) -> torch.Tensor:
    if torch.any(sigma <= 0):
        raise ValueError("sigma must be positive")
    return sigma + torch.log(-torch.expm1(-sigma))


def log_prob(
    dist: str = "Normal",
    params: dict[str, torch.Tensor] | None = None,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    if dist != "Normal" or params is None or targets is None:
        raise ValueError("only a Normal distribution with parameters and targets is supported")
    return (
        -0.5 * math.log(2 * math.pi)
        - torch.log(params["sigma"])
        - 0.5 * ((targets - params["mu"]) / params["sigma"]).square()
    )


def kl_divergence(
    dist: str = "Normal",
    params: dict[str, torch.Tensor] | None = None,
    prior_params: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if dist != "Normal" or params is None or prior_params is None:
        raise ValueError("only Normal distributions with parameters are supported")
    variance_ratio = (params["sigma"] / prior_params["sigma"]).square()
    mean_term = ((params["mu"] - prior_params["mu"]) / prior_params["sigma"]).square()
    return 0.5 * (variance_ratio + mean_term - 1 - variance_ratio.log())


__all__ = ["from_sigma", "kl_divergence", "log_prob", "to_sigma"]
