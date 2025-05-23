import matplotlib.pyplot as plt

from .plots import generate_plots
from .metrics import display_metrics
from .processing import get_aggregated_results
from utilities.meta import PLOTS_DIR, RESULTS_DIR, AVERAGE_OF, SEEDS
from utilities.fs import make_dirs


def main():
    make_dirs([PLOTS_DIR])

    displayed = [1, 2, 3, 4]

    print(f"[INFO] Displaying results only for experiments: {displayed}")
    print(f"[INFO] Averaging over these {AVERAGE_OF} seeds: {SEEDS}")

    results = get_aggregated_results(SEEDS, RESULTS_DIR, AVERAGE_OF, displayed)

    generate_plots(results, PLOTS_DIR)
    display_metrics(results)


if __name__ == "__main__":
    main()