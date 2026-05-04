[![PyPI](https://img.shields.io/pypi/v/veriflow.svg?style=flat&color=6B8E23)](https://pypi.org/project/veriflow/)
[![Docs latest](https://img.shields.io/badge/docs-latest-8B4513.svg)](https://deltares.github.io/veriflow/index.html)
[![codecov](https://codecov.io/gh/Deltares/veriflow//branch/main/graph/badge.svg)](https://codecov.io/gh/Deltares/veriflow)
[![CI](https://img.shields.io/github/actions/workflow/status/Deltares/veriflow/ci.yml?color=white&label=CI
)](https://github.com/Deltares/veriflow/actions/workflows/ci.yml)
[![Docs build](https://img.shields.io/github/actions/workflow/status/Deltares/veriflow/docs.yml?color=white&label=docs%20build)](https://github.com/Deltares/veriflow/actions/workflows/docs.yml)


# Veriflow

A verification pipeline for evaluating models and forecasts.
- 📥 Fetching data
- 🧮 Computing scores
- 📝 Writing results

## Key features
- ✅ Full control over the verification pipeline via configuration
- ✅ Native integration with [Delft-FEWS](https://oss.deltares.nl/web/delft-fews) 
- ✅ Builds on [Scores](https://scores.readthedocs.io/en/stable/) for computation of scores. This package has extensive functionality, and it's documentation is world-class.
- ✅ Extensible with your own (private) datasources, scores and datasinks
- ✅ Optimized internal datamodel for efficient computation

## Installation

Install from PyPI:

```bash
pip install veriflow
```

Or using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install veriflow
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.



## 👥 Who Is This For?

This project is aimed at anyone who's interested in assessing model and forecast quality in an easy and reproducible way, like:
- operational forecasters
- model developers
- researchers and data-scientists

## Why this package?

Verification pipelines are complex: metrics require specialized computation, data volumes may exceed memory, and sources/destinations can vary widely. This package simplifies verification by handling the entire pipeline via a single configuration file. It's reliable (tested and versioned), transparent (fully documented), reproducible, and flexible—extensible with custom datasources, scores, and datasinks. Any pipeline created is immediately transferable to other users and systems.

So wether you're working on model development or operational forecasting: this tool can help you build robust and reproducible verification pipelines.


## Technical features
- Builds on [Xarray](https://docs.xarray.dev/en/stable/#) for handling multidimensional data. 
- Supports [Zarr](https://zarr.dev/) for cloud-friendly data storage
- Supports [Dask](https://www.dask.org/) for parallel and lazy computation

