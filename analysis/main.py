import matplotlib.pyplot as plt

from .plots import generate_plots
from .metrics import display_metrics
from .processing import get_aggregated_results
from utilities.meta import plots_dir, results_dir, average_of, SEEDS
from utilities.fs import make_dirs


def main():
    make_dirs([plots_dir])

    displayed = [1, 2, 3, 4]

    print(f"[INFO] Displaying results only for experiments: {displayed}")
    print(f"[INFO] Averaging over these {average_of} seeds: {SEEDS}")

    results = get_aggregated_results(SEEDS, results_dir, average_of, displayed)

    generate_plots(results, plots_dir)
    display_metrics(results)


if __name__ == "__main__":
    main()