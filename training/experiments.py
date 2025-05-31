import torch
import copy

from .model import get_model
from .train import ce_loss
from .config import Params
from utilities.meta import DEVICE
from utilities.structs import Experiment 


def build_experiment_1_with_replay_no_decay(params: Params):
	"""
	Baseline model with replay
	"""
	model = get_model(use_decay=False)
	model.to(DEVICE)

	params = copy.deepcopy(params)
	params.decay_lambda = 0

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(1, model, params, optimizer, loss_fn, use_perfect_replay)


def build_experiment_2_no_replay_no_decay(params: Params):
	"""
	Baseline model without replay
	"""
	model = get_model(use_decay=False)
	model.to(DEVICE)

	params = copy.deepcopy(params)
	params.decay_lambda = 0

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = False

	return Experiment(2, model, params, optimizer, loss_fn, use_perfect_replay)


def build_experiment_3_no_replay_with_decay(params: Params):
	"""
	Experimental model without replay
	"""
	model = get_model(use_decay=True)
	model.to(DEVICE)

	params = copy.deepcopy(params)

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = False

	return Experiment(3, model, params, optimizer, loss_fn, use_perfect_replay)


def build_experiment_4_with_replay_with_decay(params: Params):
	"""
	Experimental model with replay
	"""
	model = get_model(use_decay=True)
	model.to(DEVICE)

	params = copy.deepcopy(params)

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(4, model, params, optimizer, loss_fn, use_perfect_replay)


def build_experiment_5_with_replay_with_decay(params: Params):
	"""
	Experimental model with replay, lower lambda
	"""
	model = get_model(use_decay=True)
	model.to(DEVICE)

	params = copy.deepcopy(params)
	params.decay_lambda = 1e-7

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(5, model, params, optimizer, loss_fn, use_perfect_replay)


def build_experiment_6_with_replay_with_decay(params: Params):
	"""
	Experimental model with replay, higher lambda
	"""
	model = get_model(use_decay=True)
	model.to(DEVICE)

	params = copy.deepcopy(params)
	params.decay_lambda = 5e-5

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(6, model, params, optimizer, loss_fn, use_perfect_replay)


def build_experiment_7_with_replay_no_decay(params: Params):
	"""
	Baseline model with replay, lower learning rate
	"""
	model = get_model(use_decay=False)
	model.to(DEVICE)

	params = copy.deepcopy(params)
	params.learning_rate = 1e-4
	params.decay_lambda = 0

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(7, model, params, optimizer, loss_fn, use_perfect_replay)


def build_experiment_8_with_replay_with_decay(params: Params):
	"""
	Experimental model with replay, lower learning rate
	"""
	model = get_model(use_decay=True)
	model.to(DEVICE)

	params = copy.deepcopy(params)
	params.learning_rate = 1e-4

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(8, model, params, optimizer, loss_fn, use_perfect_replay)


def build_experiment_9_with_replay_no_decay(params: Params):
	"""
	Baseline model with replay, higher learning rate
	"""
	model = get_model(use_decay=False)
	model.to(DEVICE)

	params = copy.deepcopy(params)
	params.learning_rate = 1e-2
	params.decay_lambda = 0

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(9, model, params, optimizer, loss_fn, use_perfect_replay)


def build_experiment_10_with_replay_with_decay(params: Params):
	"""
	Experimental model with replay, lower learning rate
	"""
	model = get_model(use_decay=True)
	model.to(DEVICE)

	params = copy.deepcopy(params)
	params.learning_rate = 1e-2

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(10, model, params, optimizer, loss_fn, use_perfect_replay)


def build_experiment_11_with_replay_no_decay(params: Params):
	"""
	Baseline model with replay, smaller model
	"""
	model = get_model(use_decay=False)
	model.to(DEVICE)

	params = copy.deepcopy(params)
	params.n_neurons = 1024
	params.decay_lambda = 0

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(11, model, params, optimizer, loss_fn, use_perfect_replay)


def build_experiment_12_with_replay_with_decay(params: Params):
	"""
	Experimental model with replay, smaller model
	"""
	model = get_model(use_decay=True)
	model.to(DEVICE)

	params = copy.deepcopy(params)
	params.n_neurons = 1024

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(12, model, params, optimizer, loss_fn, use_perfect_replay)


def build_experiment_13_with_replay_no_decay(params: Params):
	"""
	Baseline model with replay, larger model
	"""
	model = get_model(use_decay=False)
	model.to(DEVICE)

	params = copy.deepcopy(params)
	params.n_neurons = 4096
	params.decay_lambda = 0

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(13, model, params, optimizer, loss_fn, use_perfect_replay)


def build_experiment_14_with_replay_with_decay(params: Params):
	"""
	Experimental model with replay, larger model
	"""
	model = get_model(use_decay=True)
	model.to(DEVICE)

	params = copy.deepcopy(params)
	params.n_neurons = 4096

	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))
	loss_fn = ce_loss

	use_perfect_replay = True

	return Experiment(14, model, params, optimizer, loss_fn, use_perfect_replay)
