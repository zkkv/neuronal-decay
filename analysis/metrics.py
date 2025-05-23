import numpy as np


def display_metrics(results):
	print("Accuracy of task 1 before the task switch:")
	for e in results:
		print(f"Experiment {e.experiment_no}:", accuracy_before_task_switches(e.performances[0], e.switch_indices))

	print("\nAverage accuracy across all tasks (that are there at the time of computation):")
	for e in results:
		print(f"Experiment {e.experiment_no}:", average_accuracy(e.performances, e.switch_indices))

	print("\nGap depths of task 1 accuracy (p. p.):")
	for e in results:
		print(f"Experiment {e.experiment_no}:", gap_depths(e.performances[0], e.switch_indices))

	print("\nTime to recover for task 1 accuracy (batches):")
	for e in results:
		print(f"Experiment {e.experiment_no}:", time_to_recover(e.performances[0], e.switch_indices))


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
