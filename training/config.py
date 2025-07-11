from dataclasses import dataclass

from utilities.generic import str_struct


@dataclass(frozen=False)
class Params:
	"""
	Default parameters used for experiments. Each experiment can modify these if needed.
	"""
	batch_size: int         = 512
	rotations: list[int]    = (0, 80, 160)
	learning_rate: float    = 1e-3
	n_batches_per_task: int = 500
	test_size: int          = 2000
	n_neurons: int          = 2048
	decay_lambda: float     = 1e-5

	def __str__(self):
		return str_struct(self)


@dataclass(frozen=False)
class Domain:
	"""
	Domain variables for the specific dataset at hand.
	"""
	n_tasks: int        = len(Params.rotations)
	n_classes: int      = 10
	img_n_channels: int = 1
	img_size: int       = 28

	def __str__(self):
		return str_struct(self)
