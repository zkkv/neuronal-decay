import numpy as np
import warnings


def print_experiment(prefix, arr, suffix=""):
	s = f"{prefix}: "
	for a in arr:
		formatted = "[" + " ".join(f"{x:7.3f}" for x in a) + "]"
		s += f"{formatted} "
	s += suffix
	print(s)


def show_accuracy_task_1(results):
	print("\nAccuracy of task 1 before the task switch (means, stds):")
	for exps in results:
		exp_no = exps[0].experiment_no
		switch_indices = exps[0].switch_indices
		runs = [accuracy_before_task_switches(e.performances[0], switch_indices) for e in exps]
		runs = np.array(runs)
		mean = np.mean(runs, axis=0)
		std = np.std(runs, axis=0, ddof=1)
		print_experiment(f"Experiment {exp_no:02}", [mean, std])


def show_average_accuracy(results):
	print("\nAverage accuracy across all tasks (that are there at the time of computation) before the task switch (means, stds:")
	for exps in results:
		exp_no = exps[0].experiment_no
		switch_indices = exps[0].switch_indices
		runs = [average_accuracy(e.performances, switch_indices) for e in exps]
		runs = np.array(runs)
		mean = np.mean(runs, axis=0)
		std = np.std(runs, axis=0, ddof=1)
		mean_selected = accuracy_before_task_switches(mean, switch_indices)
		std_selected = accuracy_before_task_switches(std, switch_indices)
		print_experiment(f"Experiment {exp_no:02}", [mean_selected, std_selected])


def show_gap_depth(results):
	print("\nGap depths of task 1 accuracy (p. p.) (means, stds):")
	for exps in results:
		exp_no = exps[0].experiment_no
		switch_indices = exps[0].switch_indices
		runs = [gap_depths(e.performances[0], switch_indices) for e in exps]
		runs = np.array(runs)
		runs = runs[:, :, 0]  # Drop the gap index

		mean = np.mean(runs, axis=0)
		std = np.std(runs, axis=0, ddof=1)
		print_experiment(f"Experiment {exp_no:02}", [mean, std])


def show_time_to_recover(results):
	print("\nTime to recover for task 1 accuracy (%):")
	for exps in results:
		exp_no = exps[0].experiment_no
		switch_indices = exps[0].switch_indices
		runs = [time_to_recover(e.performances[0], switch_indices) for e in exps]

		runs = np.array([
			[np.nan if x is None else x for x in run]
			for run in runs
		])

		# Ignore NaN
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			mean = np.nanmean(runs, axis=0)
			std = np.nanstd(runs, axis=0, ddof=1)

			# Percentage
			mean = mean * 100 / switch_indices[0]
			std = std * 100 / switch_indices[0]
		nan_counts = np.isnan(runs).sum(axis=0)

		print_experiment(f"Experiment {exp_no:02}", [mean, std], f"Ignored {nan_counts} seeds")


def display_metrics(results):
	print("\n*METRICS*")
	show_accuracy_task_1(results)
	show_average_accuracy(results)
	show_gap_depth(results)
	show_time_to_recover(results)


def accuracy_before_task_switches(performance, switch_indices):
	'''
	Compute the accuracy at the last batch before every task switch.
	'''
	res = []
	for i in switch_indices:
		res.append(performance[i - 1])
	return res


def gap_depths(performance, switch_indices):
	'''
	Compute the depth of the deepest gap between two consecutive task switches.

	The depth is computed as:
	[accuracy before the task switch] - [accuracy at the lowest point at the task switch or after]

	Negative values indicate an increase in accuracy after the task switch.

	The result is an array of tuples with depths and the relative index of the gap (e.g. 0 is the first batch of the new task).
	'''
	accs = accuracy_before_task_switches(performance, switch_indices)
	accs.pop()  # Don't compute for the last task

	res = []
	for i, acc in enumerate(accs):
		start, end = switch_indices[i], switch_indices[i + 1]

		arg_min_per_task = np.argmin(performance[start:end])
		min_per_task = performance[arg_min_per_task + start]

		res.append((acc - min_per_task, arg_min_per_task.item()))
	return res


def time_to_recover(performance, switch_indices):
	'''
	Compute time to recover (in number of batches) from the gap to the previous accuracy level.
	Value is computed after every task switch.

		1. Find the lowest gap between two consecutive task switches.
		2. Find the first point after the gap where the accuracy is at least that at the end of the previous task.
		3. Find the difference between time values of the two.

	If the accuracy never recoveres, the resulting value is None.
	'''
	accs = accuracy_before_task_switches(performance, switch_indices)
	accs.pop()  # Don't compute for the last task

	depths = gap_depths(performance, switch_indices)

	res = []

	for i, (_, arg_gap) in enumerate(depths):
		start = switch_indices[i]
		arg_gap_absolute = arg_gap + start
		target = accs[i]
		sliced = np.array(performance[arg_gap_absolute:])

		rel_idx = np.argmax(sliced >= target) if np.any(sliced >= target) else None
		if rel_idx == None:
			res.append(None)
		else:
			ttr = (rel_idx + arg_gap).item()
			res.append(ttr)

	return res


def average_accuracy(performances, switch_indices):
	'''
	Compute average accuracy across tasks for each interval between two task switches.

	Each interval considers accuracy only of those tasks that are in progress or have been completed.
	For instance, the first interval only accounts for task 1, the second interval accounts
	for task 1 and task 2, and so on. The last interval includes all tasks in its computation.

	The result is a single list of accuracy values with the same length as performances.
	'''
	n_batches = switch_indices[-1]
	reversed_performances = [list(reversed(sublist)) for sublist in performances]
	res = []
	curr = 0

	while len(res) < n_batches:
		while curr < len(reversed_performances[-1]):
			heads = [sublist[curr] for sublist in reversed_performances]
			avg = np.mean(heads).item()
			res.append(avg)
			curr += 1
		reversed_performances.pop()

	return list(reversed(res))
