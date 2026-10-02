"""Isotropic Gaussian kernel-density estimators."""

from __future__ import annotations

import warnings

import torch

from ..utils.kde_utils import (
    IndexedDataset,
    from_pi,
    from_sigma,
    gaussian_kernel_exponent,
    to_pi,
    to_sigma,
)


def _validate_data(data: torch.Tensor, *, name: str = "data") -> None:
    if not isinstance(data, torch.Tensor) or data.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional torch tensor")
    if not data.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(data).all():
        raise ValueError(f"{name} must contain only finite values")


class VanillaKernelDensityEstimator(torch.nn.Module):
    """Kernel density estimator with one shared isotropic bandwidth."""

    def __init__(self, mu: torch.Tensor, sigma: float | torch.Tensor | None = None):
        super().__init__()
        _validate_data(mu, name="mu")
        if mu.shape[0] < 1:
            raise ValueError("mu must contain at least one row")

        self.num_kernels, self.num_dims = mu.shape
        self.register_buffer("mu", mu.detach().clone().unsqueeze(0))
        if sigma is None:
            sigma_tensor = (
                mu.std(dim=0, unbiased=False).mean() / self.num_kernels
            ).clamp_min(torch.finfo(mu.dtype).eps)
        else:
            sigma_tensor = torch.as_tensor(sigma, device=mu.device, dtype=mu.dtype)
        if sigma_tensor.numel() != 1 or sigma_tensor.item() <= 0:
            raise ValueError("sigma must be a positive scalar")

        sigma_values = sigma_tensor.expand(1, self.num_kernels, 1).clone()
        pi_values = mu.new_full((1, self.num_kernels), 1 / self.num_kernels)
        self.sigmatilde = torch.nn.Parameter(from_sigma(sigma_values), requires_grad=False)
        self.pitilde = torch.nn.Parameter(from_pi(pi_values), requires_grad=False)

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        _validate_data(x, name="x")
        exponent = gaussian_kernel_exponent(x, self.mu, to_sigma(self.sigmatilde))
        return torch.logsumexp(exponent + torch.log(to_pi(self.pitilde)), dim=1)

    def sample(
        self, num_samples: int = 1, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        kernel_index = torch.multinomial(
            to_pi(self.pitilde)[0], num_samples, replacement=True, generator=generator
        )
        selected_mu = self.mu[0, kernel_index]
        selected_sigma = to_sigma(self.sigmatilde[0, kernel_index])
        noise = torch.randn(
            num_samples,
            self.num_dims,
            device=self.mu.device,
            dtype=self.mu.dtype,
            generator=generator,
        )
        return selected_sigma * noise + selected_mu

    def expectation_step(
        self, x: torch.Tensor, leave_one_out: bool = False
    ) -> torch.Tensor:
        _validate_data(x, name="x")
        log_kernel = gaussian_kernel_exponent(x, self.mu, to_sigma(self.sigmatilde))
        log_kernel = log_kernel + torch.log(to_pi(self.pitilde))
        if leave_one_out:
            if x.shape[0] != self.num_kernels:
                raise ValueError("leave-one-out requires one input per kernel")
            log_kernel = log_kernel.masked_fill(
                torch.eye(self.num_kernels, device=x.device, dtype=torch.bool), -torch.inf
            )
        return torch.softmax(log_kernel, dim=1)

    def impute(
        self,
        x: torch.Tensor,
        num_steps: int = 100,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return a copy of ``x`` with NaNs imputed by iterative sampling."""
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or not x.is_floating_point():
            raise ValueError("x must be a two-dimensional floating-point tensor")
        if x.shape[1] != self.num_dims:
            raise ValueError("x has the wrong number of features")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")

        result = x.clone()
        missing = result.isnan()
        if not missing.any():
            return result
        result[missing] = 0
        for _ in range(num_steps):
            kernel_index = self.expectation_step(result).argmax(dim=1)
            noise = torch.randn(
                result.shape,
                device=result.device,
                dtype=result.dtype,
                generator=generator,
            )
            proposal = to_sigma(self.sigmatilde[0, kernel_index]) * noise + self.mu[
                0, kernel_index
            ]
            result[missing] = proposal[missing]
        return result

    def get_params(self) -> dict[str, torch.Tensor]:
        return {
            "mu": self.mu,
            "sigma": to_sigma(self.sigmatilde),
            "pi": to_pi(self.pitilde),
        }


class AdaptiveKernelDensityEstimator(VanillaKernelDensityEstimator):
    """KDE with one trainable isotropic bandwidth per observation."""

    def __init__(self, mu: torch.Tensor, sigma: float | torch.Tensor = 0.1):
        if mu.shape[0] < 2:
            raise ValueError("adaptive KDE requires at least two observations")
        super().__init__(mu, sigma)
        self.history_: list[float] = []
        self.n_iter_ = 0
        self.objective_ = float("nan")
        self.converged_ = False

    def forward(
        self,
        x: torch.Tensor,
        leave_one_out: bool = True,
        batch_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        exponent = gaussian_kernel_exponent(x, self.mu, to_sigma(self.sigmatilde))
        exponent = exponent + torch.log(to_pi(self.pitilde))
        if leave_one_out:
            mask = torch.eye(self.num_kernels, device=x.device, dtype=torch.bool)
            if batch_idx is not None:
                mask = mask[batch_idx]
            elif x.shape[0] != self.num_kernels:
                raise ValueError("leave-one-out requires batch indices")
            exponent = exponent.masked_fill(mask, -torch.inf)
        return torch.logsumexp(exponent, dim=1)

    def maximization_step(
        self, x: torch.Tensor, responsibility: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        squared_distance = (x.unsqueeze(1) - self.mu).square().mean(dim=-1)
        denominator = responsibility.sum(dim=0).clamp_min(torch.finfo(x.dtype).eps)
        variance = (responsibility * squared_distance).sum(dim=0) / denominator
        sigma = variance.clamp_min(torch.finfo(x.dtype).eps).sqrt().view(1, -1, 1)
        return from_sigma(sigma), self.pitilde.detach()

    def _set_trainable_parameters(self, trainable: bool) -> None:
        self.sigmatilde.requires_grad_(trainable)
        self.pitilde.requires_grad_(False)

    def _validate_fit_data(self, x: torch.Tensor, leave_one_out: bool) -> None:
        _validate_data(x, name="x")
        if x.shape != self.mu[0].shape or not torch.equal(x, self.mu[0]):
            raise ValueError("fit data must match the kernel centers supplied at construction")
        if leave_one_out and torch.unique(x, dim=0).shape[0] != x.shape[0]:
            warnings.warn(
                "repeated observations invalidate the paper's LOO singularity guarantee",
                UserWarning,
                stacklevel=2,
            )

    def fit(
        self,
        x: torch.Tensor,
        *,
        leave_one_out: bool = True,
        method: str = "em",
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        max_iterations: int = 50,
        tolerance: float | None = 1e-4,
        verbose: bool = False,
        verbose_freq: int = 10,
        generator: torch.Generator | None = None,
    ) -> "AdaptiveKernelDensityEstimator":
        """Fit bandwidths using modified EM or Adam."""
        self._validate_fit_data(x, leave_one_out)
        method = method.lower()
        if method not in {"em", "adam"}:
            raise ValueError("method must be 'em' or 'adam'")
        if max_iterations < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("iteration, batch, and learning-rate values must be positive")
        if tolerance is not None and tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        super().train(True)
        self._set_trainable_parameters(method == "adam")
        optimizer = None
        dataloader = None
        if method == "adam":
            optimizer = torch.optim.Adam(
                [parameter for parameter in self.parameters() if parameter.requires_grad],
                lr=learning_rate,
            )
            dataloader = torch.utils.data.DataLoader(
                IndexedDataset(x),
                batch_size=min(batch_size, len(x)),
                shuffle=True,
                generator=generator,
            )

        self.history_ = []
        self.converged_ = False
        previous_objective: torch.Tensor | None = None
        for iteration in range(1, max_iterations + 1):
            if method == "em":
                with torch.no_grad():
                    responsibility = self.expectation_step(x, leave_one_out)
                    sigmatilde, pitilde = self.maximization_step(x, responsibility)
                    self.sigmatilde.copy_(sigmatilde)
                    self.pitilde.copy_(pitilde)
            else:
                assert optimizer is not None and dataloader is not None
                for batch, indices in dataloader:
                    optimizer.zero_grad()
                    loss = -self(batch, leave_one_out, indices).mean()
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                objective = self(x, leave_one_out).mean()
            if not torch.isfinite(objective):
                raise ValueError("objective became non-finite")
            self.history_.append(float(objective))
            if verbose and iteration % verbose_freq == 0:
                print(f"Iteration: {iteration}, Objective Value: {objective:.5f}")
            if (
                previous_objective is not None
                and tolerance is not None
                and abs(float(objective - previous_objective)) <= tolerance
            ):
                self.converged_ = True
                break
            previous_objective = objective

        self.n_iter_ = len(self.history_)
        self.objective_ = self.history_[-1]
        self._set_trainable_parameters(False)
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
        modified_em = kwargs.pop("modified_em", False)
        batch_size = kwargs.pop("batch_size", kwargs.pop("bacth_size", 32))
        wait_convergence = kwargs.pop("wait_convergence", True)
        num_iterations = kwargs.pop("num_iterations", 50)
        objective_threshold = kwargs.pop("objective_threshold", 1e-4)
        return self.fit(
            x,
            method="em" if modified_em else "adam",
            batch_size=batch_size,
            max_iterations=1000 if wait_convergence else num_iterations,
            tolerance=objective_threshold if wait_convergence else None,
            **kwargs,
        )


class PiKernelDensityEstimator(AdaptiveKernelDensityEstimator):
    """Adaptive KDE with a trainable weight for every kernel."""

    def _set_trainable_parameters(self, trainable: bool) -> None:
        self.sigmatilde.requires_grad_(trainable)
        self.pitilde.requires_grad_(trainable)

    def maximization_step(
        self, x: torch.Tensor, responsibility: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigmatilde, _ = super().maximization_step(x, responsibility)
        pi = responsibility.sum(dim=0, keepdim=True) / responsibility.sum()
        pi = pi.clamp_min(torch.finfo(pi.dtype).tiny)
        pi = pi / pi.sum(dim=1, keepdim=True)
        return sigmatilde, from_pi(pi)


__all__ = [
    "AdaptiveKernelDensityEstimator",
    "PiKernelDensityEstimator",
    "VanillaKernelDensityEstimator",
]
