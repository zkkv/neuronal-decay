from torch.utils.data import ConcatDataset

from .data import train_datasets
from .train import train_and_eval
from utilities.meta import device, SEED, results_file
from utilities.fs import save_results_to_file
from utilities.structs import ExperimentResult


def run_experiment(experiment, domain):
	print(f" Running experiment {experiment.experiment_no}".center(60, "~"))
	performance_history = [[] for _ in range(domain.n_tasks)]
	switch_indices = []

	for task_idx in range(1, domain.n_tasks + 1):
		print(f" Training on task {task_idx} ".center(60, "="))

		if experiment.use_perfect_replay:
			dataset = ConcatDataset(train_datasets[:task_idx])
		else:
			dataset = train_datasets[task_idx - 1]

		train_and_eval(
			experiment.model,
			dataset,
			experiment.params["n_batches_per_task"],
			experiment.params["batch_size"] * task_idx,
			task_idx,
			experiment.evaluation_sets,
			experiment.params["test_size"],
			performance_history,
			experiment.optimizer,
			experiment.loss_fn,
			experiment.params["decay_lambda"],
		)
		switch_indices.append(task_idx * experiment.params["n_batches_per_task"])

	experiment.set_performance_history(performance_history)
	experiment.set_switch_indices(switch_indices)


def run_experiments(experiment_builders, domain, persist_results=True):
	results = []
	for eb in experiment_builders:
		e = eb()
		run_experiment(e, domain)

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
		save_results_to_file(results, results_file, SEED)

	print("ALL EXPERIMENTS DONE!")
	return results