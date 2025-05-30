from torch.utils.data import Dataset


class Experiment:
	'''
	A single experiment involving a set of parameters and approaches.
	'''

	def __init__(self, experiment_no, model, parameters, optimizer, loss_fn, use_perfect_replay):
		self.experiment_no = experiment_no
		self.model = model
		self.params = parameters
		self.optimizer = optimizer
		self.loss_fn = loss_fn
		self.use_perfect_replay = use_perfect_replay

		self.performances = []
		self.switch_indices = []

	def set_performance_history(self, performances):
		self.performances = performances

	def set_switch_indices(self, switch_indices):
		self.switch_indices = switch_indices

	def __repr__(self):
		attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
		return f"{self.__class__.__name__}({attrs})"


class ExperimentResult:
	'''
	A structure wrapping experiment results.
	'''

	def __init__(self, experiment_no, performances, switch_indices, parameters=None, use_perfect_replay=None, stds=None):
		self.experiment_no = experiment_no
		self.performances = performances
		self.switch_indices = switch_indices
		self.params = parameters
		self.use_perfect_replay = use_perfect_replay
		self.stds = stds

	def __repr__(self):
		attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
		return f"{self.__class__.__name__}({attrs})"


class TransformedDataset(Dataset):
	'''
	Represents a dataset with transformed value or target.
	'''

	def __init__(self, original_dataset, transform=None, target_transform=None):
		super().__init__()
		self.dataset = []
		for value, target in original_dataset:
			if transform is not None:
				value = transform(value)
			if target_transform is not None:
				target = target_transform(target)
			self.dataset.append((value, target))


	def __len__(self):
		return len(self.dataset)


	def __getitem__(self, index):
		(value, target) = self.dataset[index]
		return (value, target)


class CircularIterator:
	def __init__(self, iterable):
		self.items = list(iterable)
		if not self.items:
			raise ValueError("Empty iterable")
		self.index = 0

	def __iter__(self):
		return self

	def __next__(self):
		value = self.items[self.index]
		self.index = (self.index + 1) % len(self.items)
		return value
