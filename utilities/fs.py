import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime

from utilities.structs import ExperimentResult


def save_results_to_file(results, results_file, domain, train_len, test_len, seed, should_log=True):
	mapped = {}

	for res in results:
		# Note: We save most values just for logging purposes.
		#       Not everything is used for evaluation/visualization.
		domain_vars = {
			"n_tasks": domain.n_tasks,
			"n_classes": domain.n_classes,
			"img_n_channels": domain.img_n_channels,
			"img_size": domain.img_size,
			"len(training_data)": train_len,
			"len(test_data)": test_len,
		}

		mapped_experiment = {
			"experiment_no": res.experiment_no,
			"performances": res.performances,
			"switch_indices": res.switch_indices,
			"parameters": res.params,
			"use_perfect_replay": res.use_perfect_replay,
			"domain_variables": domain_vars,
			"seed": seed,
		}
		mapped[f"{res.experiment_no}"] = mapped_experiment

	with open(results_file, 'w') as f:
		if should_log:
			print(f"[INFO] Saving results to {results_file}")
		json.dump(mapped, f, indent=2)


def load_results_from_file(results_file, experiments, should_log=True):
	results = []
	with open(results_file, 'r') as f:
		if should_log:
			print(f"[INFO] Reading results from {results_file}")
		obj = json.load(f)

	if len(experiments) == 0:
		str_keys = list(obj.keys())
		int_keys = sorted([int(e) for e in str_keys])
		experiments.extend(int_keys)

	for res_no in experiments:
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


@contextmanager
def tee(is_quiet, logfile_path, should_log_time=True):
	is_logging = logfile_path is not None
	old_stdout = sys.stdout

	if is_logging:
		logfile = open(logfile_path, 'a')
		if should_log_time:
			logfile.write(str(datetime.now().isoformat()) + "\n")
	else:
		logfile = None

	class Stream:
		def write(self, message):
			if not is_quiet:
				old_stdout.write(message)
			if is_logging:
				logfile.write(message)
		def flush(self):
			if not is_quiet:
				old_stdout.flush()
			if is_logging:
				logfile.flush()

	sys.stdout = Stream()

	try:
		yield
	finally:
		sys.stdout = old_stdout
		if is_logging:
			logfile.close()

