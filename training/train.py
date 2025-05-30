import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from thop import profile

from .model import macs_decay
from utilities.meta import DEVICE
from utilities.structs import CircularIterator


def ce_loss(prediction, y, reduction='mean'):
	return F.cross_entropy(input=prediction, target=y, reduction=reduction)


def compute_accuracy(model, dataset, test_size=None, batch_size=128):
	'''
	Compute accuracy (% samples classified correctly) of a classifier ([model]) on [dataset].
	'''

	mode = model.training
	model.eval()

	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
	total_tested = 0
	total_correct = 0

	for X, y in dataloader:
		if test_size and total_tested >= test_size:
			break

		X, y = X.to(DEVICE), y.to(DEVICE)
		with torch.no_grad():
			scores, _ = model(X)
		_, pred = torch.max(scores, 1)
		total_correct += (pred == y).sum().item()
		total_tested += len(X)
	accuracy = total_correct * 100 / total_tested

	model.train(mode=mode)

	return accuracy


def train_and_eval(experiment, train_set, test_sets, task_idx, performance, is_profiling):
	'''
	Function to train a [model] on a given [train_set],
	while evaluating after each training iteration on [test_sets].
	'''
	model = experiment.model
	n_batches = experiment.params.n_batches_per_task
	batch_size = experiment.params.batch_size * task_idx
	test_size = experiment.params.test_size
	optimizer = experiment.optimizer
	loss_fn = experiment.loss_fn
	decay_lambda = experiment.params.decay_lambda

	model.train()
	print_every_n = 25

	dataloader = CircularIterator(DataLoader(train_set, batch_size=batch_size, shuffle=True))

	for batch_idx in range(n_batches):

		X, y = next(dataloader)
		X, y = X.to(DEVICE), y.to(DEVICE)

		# Prediction
		pred, decay = model(X)

		# Evaluation
		loss = loss_fn(pred, y) + decay_lambda * decay

		# Profiling
		if is_profiling:
			macs, _ = profile(model, inputs=(X,), verbose=False, custom_ops={type(model): macs_decay})

		for test_idx, test_set in enumerate(test_sets):
			if test_idx >= task_idx:
				break
			accuracy = compute_accuracy(model, test_set, test_size, batch_size)
			performance[test_idx].append(accuracy)

		# Backpropagation
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		# Logging
		if batch_idx % print_every_n == 0:
			log_str = ('Training loss: {loss:04f} | Test accuracy (task 1): {prec:05.2f}% | Batch: {b_index}'
				.format(loss=loss.item(), prec=performance[0][-1], b_index=batch_idx))
			if is_profiling:
				log_str += f' | MACs: {macs}'
			print(log_str)