from . import experiments
from .execution import run_experiments 
from .config import Domain, Params
from .cli import parse_args
from utilities.meta import DATA_DIR, OUT_DIR, RESULTS_DIR, DEVICE
from utilities.fs import make_dirs


def main():
	argv = parse_args()
	seed = argv.seed
	decay_lambda = argv.lam

	make_dirs([DATA_DIR, OUT_DIR, RESULTS_DIR])

	domain = Domain()
	params = Params()

	if decay_lambda is not None:
		params.decay_lambda = decay_lambda

	print(f"[INFO] Using device: {DEVICE}")
	print(f"[INFO] Data directory: {DATA_DIR}, Results directory: {RESULTS_DIR}")
	print(f"[INFO] Hyperparameters: {params}")
	print(f"[INFO] Domain variables: {domain}")

	experiment_builders = [
		experiments.build_experiment_1_with_replay_no_decay,
		experiments.build_experiment_2_no_replay_no_decay,
		experiments.build_experiment_3_no_replay_with_decay,
		experiments.build_experiment_4_with_replay_with_decay,
	]

	run_experiments(experiment_builders, params, domain, seed)


if __name__ == "__main__":
	main()
