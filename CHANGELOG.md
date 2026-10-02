# Changelog

## 0.2.0 - Unreleased

### Fixed

- Correct the second within-sample coefficient in the energy-distance statistic.
- Restore standard PyTorch `train(mode)` and `.eval()` behavior.
- Register KDE parameters and VAE priors for device movement and serialization.
- Use tensor-aware device and dtype handling throughout sampling and fitting.
- Bound KDE optimization and expose convergence information.
- Avoid modifying tensors passed to imputation methods.
- Honor explicitly supplied VAE prior parameters in KL divergence.

### Added

- Conventional `fit()` methods with deprecated 0.1.x training shims.
- Explicit random-generator arguments for reproducible stochastic operations.
- Input validation and a warning when duplicate rows invalidate the LOO guarantee.
- Public package exports, regression tests, citation metadata, and reproducibility documentation.

### Changed

- Require Python 3.10 or newer and remove the old Python 3.10 upper bound.
- Mark VAE functionality as experimental and outside the paper's validated KDE contribution.
