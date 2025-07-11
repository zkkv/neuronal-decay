from .cli import parse_args
from .plots import generate_plots
from .metrics import display_metrics
from .processing import load_results, aggregate_results, transpose_runs
from utilities.meta import PLOTS_DIR, RESULTS_DIR, LOG_DIR
from utilities.fs import make_dirs, tee


def run(seeds, displayed, format):
	print(f"[INFO] Plots directory; {PLOTS_DIR}, Results directory: {RESULTS_DIR}, Logs directory: {LOG_DIR}")
	if len(displayed) > 0:
		print(f"[INFO] Displaying results for experiments: {displayed}")
	else:
		print(f"[INFO] Displaying results for all experiments")

	if len(seeds) == 0:
		print(f"[INFO] Using results without seed")
	elif len(seeds) == 1:
		print(f"[INFO] Using seed: {seeds[0]}")
	else:
		print(f"[INFO] Averaging over these {len(seeds)} seeds: {seeds}")

	runs = load_results(seeds, RESULTS_DIR, displayed)
	results_visuals = aggregate_results(runs, displayed)
	results = transpose_runs(runs, displayed)

	generate_plots(results_visuals, PLOTS_DIR, format)
	display_metrics(results)


def main():
	argv = parse_args()
	seeds = argv.seeds
	displayed = argv.display
	is_quiet = argv.quiet
	is_logging = not argv.no_log
	format = "pdf"

	make_dirs([PLOTS_DIR])

	logfile_path = f"{LOG_DIR}/analysis.log" if is_logging else None

	with tee(is_quiet, logfile_path, should_log_time=True):
		run(seeds, displayed, format)


if __name__ == "__main__":
	main()
