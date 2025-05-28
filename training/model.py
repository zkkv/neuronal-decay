import torch
from torch import nn
import torch.nn.functional as F

from .config import Domain, Params


def get_model(use_decay):
	return Classifier(
		Domain.img_size,
		Domain.img_n_channels,
		Domain.n_classes,
		Params.n_neurons,
		use_decay,
	)


def macs_decay(model, X, y):
	if not isinstance(model, Classifier):
		raise NotImplementedError(f"Logic for counting MACs is not defined for type {type(model)}")

	if not model.use_decay:
		return torch.DoubleTensor([0])

	batch_size = X[0].shape[0]
	total = 0
	total += model.fc1.out_features * batch_size
	total += model.fc2.out_features * batch_size
	model.total_ops = torch.DoubleTensor([total])


class Classifier(nn.Module):
	def __init__(self, img_size, img_n_channels, n_classes, n_neurons, use_decay):
		super().__init__()
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(img_n_channels * img_size * img_size, n_neurons)
		self.fc2 = nn.Linear(n_neurons, n_neurons)
		self.fc3 = nn.Linear(n_neurons, n_classes)
		self.use_decay = use_decay


	def forward(self, x):
		x = self.flatten(x)
		l2_decay = 0.0

		a1 = F.relu(self.fc1(x))
		if self.use_decay:
			l2_decay += torch.sum(a1**2)

		a2 = F.relu(self.fc2(a1))
		if self.use_decay:
			l2_decay += torch.sum(a2**2)

		logits = self.fc3(a2)

		return logits, l2_decay
