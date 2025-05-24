import torch

from .model import get_model
from .train import ce_loss
from .config import Params
from utilities.meta import DEVICE
from utilities.structs import Experiment 


def build_experiment_1_with_replay_no_decay(params: Params):
	model = get_model()
	model.to(DEVICE)

	params_dict = {
		"batch_size": params.batch_size,
		"rotations": params.rotations,
		"learning_rate": params.learning_rate,
		"n_batches_per_task": params.n_batches_per_task,
		"test_size": params.test_size,
		"decay_lambda": 0,
	}

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(1, model, params_dict, optimizer, loss_fn, use_perfect_replay)


def build_experiment_2_no_replay_no_decay(params: Params):
	model = get_model()
	model.to(DEVICE)

	params_dict = {
		"batch_size": params.batch_size,
		"rotations": params.rotations,
		"learning_rate": params.learning_rate,
		"n_batches_per_task": params.n_batches_per_task,
		"test_size": params.test_size,
		"decay_lambda": 0,
	}

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = False

	return Experiment(2, model, params_dict, optimizer, loss_fn, use_perfect_replay)


def build_experiment_3_no_replay_with_decay(params: Params):
	model = get_model()
	model.to(DEVICE)

	params_dict = {
		"batch_size": params.batch_size,
		"rotations": params.rotations,
		"learning_rate": params.learning_rate,
		"n_batches_per_task": params.n_batches_per_task,
		"test_size": params.test_size,
		"decay_lambda": params.decay_lambda,
	}

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = False

	return Experiment(3, model, params_dict, optimizer, loss_fn, use_perfect_replay)


def build_experiment_4_with_replay_with_decay(params: Params):
	model = get_model()
	model.to(DEVICE)

	params_dict = {
		"batch_size": params.batch_size,
		"rotations": params.rotations,
		"learning_rate": params.learning_rate,
		"n_batches_per_task": params.n_batches_per_task,
		"test_size": params.test_size,
		"decay_lambda": params.decay_lambda,
	}

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(4, model, params_dict, optimizer, loss_fn, use_perfect_replay)
