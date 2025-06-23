# Neuronal Decay

The experimental code for the paper "I Fought the Low: Decreasing Stability Gap with Neuronal Decay".

## Prerequisites

1. Get Python 3.12 (or above).
2. Install dependencies in one of two ways:
   - get a Python build tool called [poetry](https://python-poetry.org/).
   - install all dependencies manually, e.g. via `pip`. The versions of all dependencies are specified in the `pyproject.toml` file.

If you want to reproduce the results, using the same versions of packages and Python itself is recommended. For that reason a `poetry.lock` file is provided. If you have any issues when installing the dependencies, first try to delete that file.

> [!TIP]
> By default, PyTorch installs all necessary CUDA packages. If you want to run PyTorch code on the CPU and don't need these extra packages, you can replace the source repository of `torch` and `torchvision` to `torch-cpu` in `pyproject.toml` before proceeding to the next step.

## Quick Start
Install all dependencies:
```shell
poetry install
```

Confirm all dependencies are found (optional):
```shell
./import_test.sh
```

If you don't see any error messages, you can safely run the experiments:
```shell
./run.sh
```

This script runs the experiments with selected seeds. Then it runs the analysis code that calculates metrics and generates plots. You can find all outputs in the `out` directory.

If you want to run a smaller set of experiments with profiler enabled and no further analysis, execute:
```shell
./run_profile.sh
```

## Artifacts

The results, logs and plots from my own past runs are stored in the `artifacts` directory. No new results are saved there, but if you just want to take a look at the data, you can find it in this directory.

## Extended Usage

The shell scripts contain a single reproducible setup, but you can customize some parts of the program using the CLI. The most basic way to run the training code is:
```shell
poetry run python -m training.main
```

This will run the experiments with a random seed. However, you may want, for instance, to specify the seed. In this case, pass the seed after the `-s` option. You can also overwrite the neuronal decay lambda (for some experiments) with the `-l` option. More details can be found by using `--help`.

The analysis program can be configured too. You can pass specific seeds that you want to use to the core command:
```shell
poetry run python -m analysis.main
```

Learn more by using `--help`.

> [!IMPORTANT]
> The analysis code looks for results in the `out/results` directory and uses the file names to match them with the provided seeds. Make sure to keep this directory clear of any other files. 


## License
All code in this repository produced by me is licensed under the [MIT License](https://choosealicense.com/licenses/mit/). All dependencies are licensed under their respective licenses.

## Developer
Developed by Kirill Zhankov, 2025
