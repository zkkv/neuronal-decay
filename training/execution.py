from torch.utils.data import ConcatDataset
import numpy as np
import torch

from .train import train_and_eval
from utilities.meta import RESULTS_DIR, DEVICE
from utilities.fs import save_results_to_file
from utilities.structs import ExperimentResult
from utilities.profiling import optional_profiler, get_total_time


def run_experiment(experiment, domain, train_datasets, test_datasets, is_profiling):
	print(f" Running experiment {experiment.experiment_no}".center(60, "~"))
	performance_history = [[] for _ in range(domain.n_tasks)]
	switch_indices = []

	for task_idx in range(1, domain.n_tasks + 1):
		print(f" Training on task {task_idx} ".center(60, "="))

		if experiment.use_perfect_replay:
			dataset = ConcatDataset(train_datasets[:task_idx])
		else:
			dataset = train_datasets[task_idx - 1]

		with optional_profiler(is_profiling, DEVICE) as prof:
			train_and_eval(
				experiment,
				dataset,
				test_datasets,
				task_idx,
				performance_history,
				is_profiling,
			)
			if is_profiling:
				prof.step()

		true_n_batches = task_idx * experiment.params.n_batches_per_task
		switch_indices.append(true_n_batches)

		if is_profiling:
			cpu_time_ms, cuda_time_ms = get_total_time(prof.key_averages())
			print(f"Self CPU time per batch (ms):", cpu_time_ms / true_n_batches)

			if DEVICE == "cuda":
				print(f"Self CUDA time per batch (ms):", cuda_time_ms / true_n_batches)

	experiment.set_performance_history(performance_history)
	experiment.set_switch_indices(switch_indices)


def run_experiments(
		experiment_builders,
		params,
		domain,
		train_datasets,
		test_datasets,
		seed,
		is_profiling=False,
		persist_results=True):
	is_deterministic = True if seed is not None else False

	if is_deterministic:
		np.random.seed(seed)
		torch.manual_seed(seed)
	else:
		print("[WARN] No seed was set")

	results = []
	for eb in experiment_builders:
		e = eb(params)
		run_experiment(e, domain, train_datasets, test_datasets, is_profiling)

		res = ExperimentResult(
			e.experiment_no,
			e.performances,
			e.switch_indices,
			parameters=e.params,
			use_perfect_replay=e.use_perfect_replay
		)
		results.append(res)
		print(f"Experiment {res.experiment_no} done!")

	if persist_results:
		if seed is None:
			results_file = f"{RESULTS_DIR}/results.json"
		else:
			results_file = f"{RESULTS_DIR}/results_{seed}.json"
		train_len, test_len = len(train_datasets[0]), len(test_datasets[0])
		save_results_to_file(results, results_file, domain, train_len, test_len, seed)

	print("ALL EXPERIMENTS DONE!")
	return results
