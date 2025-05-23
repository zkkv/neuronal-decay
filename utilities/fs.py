import json
import os

from training.config import Domain, Params  # FIXME: should not depend on training module
from training.data import training_data, test_data
from utilities.structs import ExperimentResult


def save_results_to_file(results, results_file, seed, should_log=True):
	mapped = {}

	for res in results:
		# Note: We save most values just for logging purposes.
		#       Not everything is used for evaluation/visualization.
		domain_vars = {
			"n_tasks": Domain.n_tasks,
			"n_classes": Domain.n_classes,
			"img_n_channels": Domain.img_n_channels,
			"img_size": Domain.img_size,
			"len(training_data)": len(training_data),
			"len(test_data)": len(test_data),
		}

		mapped_experiment = {
			"experiment_no": res.experiment_no,
			"performances": res.performances,
			"switch_indices": res.switch_indices,
			"parameters": res.params,
			"use_perfect_replay": res.use_perfect_replay,
			"domain_variables": domain_vars,
			"seed": seed,
			"n_neurons": Params.n_neurons,
		}
		mapped[f"{res.experiment_no}"] = mapped_experiment

	with open(results_file, 'w') as f:
		if should_log:
			print(f"[INFO] Saving results to {results_file}")
		json.dump(mapped, f, indent=2)


def load_results_from_file(results_file, displayed, should_log=True):
	results = []
	with open(results_file, 'r') as f:
		if should_log:
			print(f"[INFO] Reading results from {results_file}")
		obj = json.load(f)

	for res_no in displayed:
		res_obj = obj.get(str(res_no), None)
		if res_obj == None:
			continue

		res = ExperimentResult(
			res_obj['experiment_no'],
			res_obj['performances'],
			res_obj['switch_indices'],
		)
		results.append(res)

	results = sorted(results, key=lambda e: e.experiment_no)
	return results


def make_dirs(dirs):
	for directory in dirs:
		os.makedirs(directory, exist_ok=True)