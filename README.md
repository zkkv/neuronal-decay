# Neuronal Decay

## Prerequisites

1. Get Python 3.12 (or above).
2. Make sure to get a Python build tool called [poetry](https://python-poetry.org/). You'll need it in order to install all necessary dependecies. You can also install all dependencies manually, e.g. via `pip`. The versions of all dependencies are specified in the `pyproject.toml` file.

If you want to reproduce the results, using the same versions of packages and Python itself is recommended.

Tip: By default, PyTorch installs necessary CUDA packages. If you want to run PyTorch code on the CPU and don't need all these extra packages, you can replace the source repository of `torch` and `torchvision` to `torch-cpu` in `pyproject.toml` before proceeding to the next step.

## Quick Start
Install all dependencies:
```
poetry install
```

Confirm all dependencies are found (optional):
```
./import_test.sh
```

If you don't see any error messages, you can safely run the experiments:
```
./run.sh
```

This script runs the experiments with selected seeds. Then it runs the analysis code that calculates metrics and generates plots. You can find all outputs in the `out/` directory.
