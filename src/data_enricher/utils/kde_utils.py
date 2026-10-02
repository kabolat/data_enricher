"""Numerical helpers for KDE models."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def to_sigma(sigmatilde: torch.Tensor) -> torch.Tensor:
    log_two = math.log(2)
    return F.softplus(sigmatilde * log_two) / log_two


def from_sigma(sigma: torch.Tensor) -> torch.Tensor:
    if torch.any(sigma <= 0):
        raise ValueError("sigma must be positive")
    log_two = math.log(2)
    return sigma + torch.log(-torch.expm1(-sigma * log_two)) / log_two


def to_pi(pitilde: torch.Tensor) -> torch.Tensor:
    return torch.softmax(pitilde, dim=1)


def from_pi(pi: torch.Tensor) -> torch.Tensor:
    if torch.any(pi <= 0):
        raise ValueError("pi must be positive")
    return torch.log(pi) - torch.log(pi).mean(dim=1, keepdim=True)


def gaussian_kernel_exponent(
    x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Return Gaussian log densities for every input/kernel pair."""
    dimensions = mu.shape[-1]
    squared_distance = (x.unsqueeze(1) - mu).square().sum(dim=-1)
    variance = sigma.squeeze(-1).square()
    return -0.5 * (
        squared_distance / variance
        + dimensions * torch.log(variance)
        + dimensions * math.log(2 * math.pi)
    )


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.data[index], index

    def __len__(self) -> int:
        return len(self.data)


__all__ = [
    "IndexedDataset",
    "from_pi",
    "from_sigma",
    "gaussian_kernel_exponent",
    "to_pi",
    "to_sigma",
]
