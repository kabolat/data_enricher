# Reproducibility guide

Data Enricher has two related but distinct roles: it is an installable Python
package, and it contains implementations related to a published research
result. The resources below preserve those roles without rewriting history.

## Exact paper-era package

The immutable `paper-v1` tag points to commit
`f801558fa18d34ce9e1d6d5b1b4a775dc1c3c4d6`, the package version identified by
the author as the version used for the paper.

```bash
git clone --branch paper-v1 --single-branch \
  https://github.com/kabolat/data_enricher.git data_enricher-paper
cd data_enricher-paper
python -m pip install -e .
```

The saved notebooks record Python 3.9.15. The package metadata requires
PyTorch 2.0 or newer and SciPy 1.8.1 or newer, but an exact transitive lockfile
was not recorded in 2023. The tag therefore preserves source code exactly,
not a bit-for-bit environment.

## Full paper experiments

The publication's Europe and Denmark experiments are maintained separately at
[`kabolat/leave-one-out_maximum-log-likelihood`](https://github.com/kabolat/leave-one-out_maximum-log-likelihood).
That repository is the source of truth for:

- the curated Europe and Denmark datasets;
- GMM benchmark implementations;
- experiment and visualization notebooks; and
- saved outputs used to prepare the paper's figures and tables.

The experiment notebooks use `torch-two-sample`'s `MMDStatistic` and
`EnergyStatistic`. They do **not** use the `EnergyTest` function that shipped
in Data Enricher 0.1.3. Consequently, the incorrect energy coefficient fixed
in Data Enricher 0.2 affected the package's demonstration utility, not the
published paper's energy-test results.

## Maintained package

The default branch targets current users. Version 0.2 fixes statistical and
PyTorch integration bugs, adds explicit random generators, validates the
paper's assumptions, and retains deprecated compatibility shims for 0.1.x.
These corrections intentionally mean that new results can differ from results
produced by the historical package.

For a quick deterministic check of the paper's central stability intuition:

```bash
python examples/loo_stability.py
```

For new studies, record at least:

- the Data Enricher version and Git commit;
- Python, PyTorch, SciPy, hardware, and accelerator versions;
- all random seeds or serialized generator states;
- data preprocessing and train/test splits; and
- fitting method, convergence tolerance, and maximum iterations.

## Data provenance

The historical `examples/data/europe.csv` is derived from ENTSO-E load data
distributed through the Open Power System Data time-series package. See the
paper and full experiment repository for the country selection, years,
normalization, and original source links. Confirm the upstream data terms
before redistributing derived datasets outside this repository.
