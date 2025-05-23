import matplotlib.pyplot as plt

from .cli import parse_args
from .plots import generate_plots
from .metrics import display_metrics
from .processing import get_aggregated_results
from utilities.meta import PLOTS_DIR, RESULTS_DIR
from utilities.fs import make_dirs, quiet_mode


def run(seeds, displayed):
	if len(displayed) > 0:
		print(f"[INFO] Displaying results for experiments: {displayed}")
	else:
		print(f"[INFO] Displaying results for all experiments")

	if len(seeds) > 1:
		print(f"[INFO] Averaging over these {len(seeds)} seeds: {seeds}")
	else:
		print(f"[INFO] Using seed: {seeds[0]}")

	results = get_aggregated_results(seeds, RESULTS_DIR, displayed)

	generate_plots(results, PLOTS_DIR)
	display_metrics(results)


def main():
	argv = parse_args()
	seeds = argv.seeds
	displayed = argv.display
	is_quiet = argv.quiet

	make_dirs([PLOTS_DIR])

	if is_quiet:
		with quiet_mode():
			run(seeds, displayed)
	else:
		run(seeds, displayed)


if __name__ == "__main__":
	main()