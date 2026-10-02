# Data Enricher

Data Enricher is a small PyTorch package for probabilistic modelling of tabular
data. It provides isotropic Gaussian kernel-density estimators (KDEs) for data
generation and missing-value imputation, together with statistical comparison
utilities.

The adaptive KDE and π-KDE implementations accompany the paper
[Stable training of probabilistic models using the leave-one-out maximum
log-likelihood objective](https://doi.org/10.1016/j.epsr.2024.110775) by Bölat,
Tindemans, and Palensky (2024).

> The VAE implementation is experimental. It is useful for comparison and
> exploration, but it is not part of the KDE method or guarantees established
> by the paper.

## Installation

Data Enricher requires Python 3.10 or newer.

```bash
pip install data-enricher
```

To work from a clone:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Quick start

```python
import torch

from data_enricher import PiKernelDensityEstimator

data = torch.tensor(
    [
        [-1.2, 0.3],
        [-0.7, 0.1],
        [0.2, 0.8],
        [0.9, 1.1],
    ]
)

model = PiKernelDensityEstimator(data, sigma=0.2)
model.fit(data, method="em")

generator = torch.Generator().manual_seed(42)
generated = model.sample(100, generator=generator)
log_likelihood = model.log_likelihood(data)
```

KDE centers are the observations supplied at construction. Consequently, the
data passed to `fit` must be the same tensor values. Available models are:

- `VanillaKernelDensityEstimator`: fixed, shared bandwidth and uniform weights.
- `AdaptiveKernelDensityEstimator`: one fitted bandwidth per observation.
- `PiKernelDensityEstimator`: fitted bandwidths and kernel weights.

Adaptive models support the paper's modified EM algorithm and Adam:

```python
model.fit(data, method="em", max_iterations=50, tolerance=1e-4)
model.fit(data, method="adam", learning_rate=1e-2, batch_size=32)
```

After fitting, `converged_`, `n_iter_`, `objective_`, and `history_` expose the
optimization outcome. All model tensors are included in `state_dict`, and the
usual PyTorch `.to(...)`, `.train()`, and `.eval()` methods work normally.

## Imputation

`impute` returns a new tensor and does not modify its input:

```python
incomplete = torch.tensor([[float("nan"), 0.3], [0.4, 0.8]])
completed = model.impute(
    incomplete,
    num_steps=100,
    generator=torch.Generator().manual_seed(42),
)
```

## Comparing generated samples

```python
from data_enricher import model_comparison, sample_comparison

model_scores, baseline_scores = sample_comparison(
    model_samples=generated,
    train_samples=data,
    test_samples=test_data,
    test="mmd",              # or "energy"
    subsample_ratio=0.4,
    mc_runs=1000,
    generator=torch.Generator().manual_seed(42),
)

score = model_comparison(model_scores, baseline_scores, test="ks")
```

`model_comparison` also supports `"cvm"` and `"mean"`. Smaller KS and CvM
scores indicate closer distributions of sample-comparison statistics.

## Scientific assumptions and limits

- The LOO singularity-prevention guarantee assumes no repeated observations.
  Fitting with exact duplicate rows emits a warning because the model can still
  be useful, but the theorem no longer applies.
- Training and comparison use pairwise matrices and therefore require
  quadratic memory in the number of observations. The implementation targets
  research-scale tabular datasets rather than very large datasets.
- KDE kernels are isotropic. Data on a low-dimensional manifold can therefore
  produce noisy samples, as discussed in the paper.
- Inputs must be finite, two-dimensional floating-point tensors. Standardize
  features when their scales differ materially.
- Reproducibility requires an explicitly seeded `torch.Generator`; GPU results
  can still vary across hardware and PyTorch builds.

## Compatibility with 0.1.x

Version 0.2 retains the documented 0.1.x entry points while correcting their
behavior:

- `model.train(x=...)` still fits a model but emits `DeprecationWarning`; use
  `model.fit(x, ...)`. PyTorch's normal `train(mode)` and `.eval()` now work.
- `MMDTest` and `EnergyTest` remain aliases for `mmd_test` and `energy_test`.
- The misspelled KDE argument `bacth_size` is accepted by the deprecated
  training call; new code should use `batch_size`.

These shims are intended to be removed in version 1.0.

## Paper reproduction

The exact package snapshot associated with the paper is permanently tagged
[`paper-v1`](https://github.com/kabolat/data_enricher/tree/paper-v1). Full
Europe and Denmark experiment notebooks, data, and saved outputs live in the
separate
[`leave-one-out_maximum-log-likelihood`](https://github.com/kabolat/leave-one-out_maximum-log-likelihood)
repository.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the distinction between the
archived paper software, the full experiment repository, and the maintained
package.

## Citation

Please cite the journal article when the KDE method contributes to published
work:

```bibtex
@article{bolat2024stable,
  title   = {Stable training of probabilistic models using the leave-one-out maximum log-likelihood objective},
  author  = {B{\"o}lat, Kutay and Tindemans, Simon H. and Palensky, Peter},
  journal = {Electric Power Systems Research},
  volume  = {235},
  pages   = {110775},
  year    = {2024},
  doi     = {10.1016/j.epsr.2024.110775}
}
```

GitHub and Zenodo-compatible metadata are also provided in
[`CITATION.cff`](CITATION.cff).

## License and acknowledgement

The software is distributed under the [MIT License](LICENSE). Development was
supported by the EU Horizon 2020 InnoCyPES project under Marie Skłodowska-Curie
grant agreement No. 956433.
