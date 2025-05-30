from . import experiments
from .execution import run_experiments
from .config import Domain, Params
from .cli import parse_args
from .data import get_datasets
from utilities.meta import DATA_DIR, OUT_DIR, RESULTS_DIR, LOG_DIR, DEVICE
from utilities.fs import make_dirs, tee


def run(params, domain, seed, is_profiling):
	print(f"[INFO] Getting and transforming the data...")
	train_datasets, test_datasets = get_datasets(DATA_DIR, params.rotations)

	print(f"[INFO] Training set size = {len(train_datasets[0])}, Test set size = {len(test_datasets[0])}")
	print(f"[INFO] Using device: {DEVICE}")
	print(f"[INFO] Data directory: {DATA_DIR}, Results directory: {RESULTS_DIR}, Logs directory: {LOG_DIR}")
	print(f"[INFO] Hyperparameters: {params}")
	print(f"[INFO] Domain variables: {domain}")

	experiment_builders = [
		experiments.build_experiment_1_with_replay_no_decay,
		experiments.build_experiment_2_no_replay_no_decay,
		experiments.build_experiment_3_no_replay_with_decay,
		experiments.build_experiment_4_with_replay_with_decay,
		experiments.build_experiment_5_with_replay_with_decay,
		experiments.build_experiment_6_with_replay_with_decay,
		experiments.build_experiment_7_with_replay_no_decay,
		experiments.build_experiment_8_with_replay_with_decay,
		experiments.build_experiment_9_with_replay_no_decay,
		experiments.build_experiment_10_with_replay_with_decay,
		experiments.build_experiment_11_with_replay_no_decay,
		experiments.build_experiment_12_with_replay_with_decay,
		experiments.build_experiment_13_with_replay_no_decay,
		experiments.build_experiment_14_with_replay_with_decay,
	]

	run_experiments(
		experiment_builders,
		params,
		domain,
		train_datasets,
		test_datasets,
		seed,
		is_profiling=is_profiling,
	)


def main():
	argv = parse_args()
	seed = argv.seed
	decay_lambda = argv.lam
	is_quiet = argv.quiet
	is_logging = not argv.no_log
	is_profiling = argv.profile

	make_dirs([DATA_DIR, OUT_DIR, RESULTS_DIR, LOG_DIR])

	domain = Domain()
	params = Params()

	if decay_lambda is not None:
		params.decay_lambda = decay_lambda

	logfile_path = f"{LOG_DIR}/training.log" if is_logging else None

	with tee(is_quiet, logfile_path, should_log_time=True):
		run(params, domain, seed, is_profiling)


if __name__ == "__main__":
	main()
