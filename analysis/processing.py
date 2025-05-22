from utilities.structs import ExperimentResult
from utilities.generic import average_inhomogeneous
from utilities.fs import load_results_from_file, make_dirs


def get_matching(runs, exp_no):
	collected = []
	for run in runs:
		matching = list(filter(lambda x: x.experiment_no == exp_no, run))
		if len(matching) == 0:
			return None
		collected.append(matching[0])
	return collected


def get_aggregated_results(seeds, results_dir, average_of, displayed):
	'''
	Average results for each experiment across multiple runs.

	Each run is stored in a separate JSON file and is named "results_SEED.json" where SEED 
	refers to the seed of the run.

	To avoid issues, all JSON files should have the same format and
	the experiments should have the same experimental setups, i.e. only the seed should change.
	'''

	if average_of == 1:
		print("[WARN] Using only one seed, so sample SD won't be computed")

	# Load every JSON results file from each run
	runs = []
	for seed in seeds:
		results_file = f"{results_dir}/results_{seed}.json"
		results_from_file = load_results_from_file(results_file, displayed)
		results_from_file = list(filter(lambda x: x.experiment_no in displayed, results_from_file))
		runs.append(results_from_file)

	# Average results from one or more JSON files per experiment
	aggregated = []
	for exp_no in displayed:
		# Collect results for a single experiment from all files
		collected = get_matching(runs, exp_no)
		if collected is None:
			continue

		# Average the results and save it as a single experiment
		avg_performances, stds = average_inhomogeneous([c.performances for c in collected], compute_std=(average_of >= 2))
		res = ExperimentResult(
			collected[0].experiment_no,
			avg_performances,
			collected[0].switch_indices,
			parameters=collected[0].params,
			use_perfect_replay=collected[0].use_perfect_replay,
			stds=stds,
		)
		aggregated.append(res)

	return aggregated