import torch
from torch import nn
import torch.nn.functional as F

from .config import Domain, Params


def get_model():
	return Classifier(
		Domain.img_size,
		Domain.img_n_channels,
		Domain.n_classes,
		Params.n_neurons
	)


class Classifier(nn.Module):
	def __init__(self, img_size, img_n_channels, n_classes, n_neurons):
		super().__init__()
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(img_n_channels * img_size * img_size, n_neurons)
		self.fc2 = nn.Linear(n_neurons, n_neurons)
		self.fc3 = nn.Linear(n_neurons, n_classes)


	def forward(self, x):
		x = self.flatten(x)
		l2_decay = 0

		a1 = F.relu(self.fc1(x))
		l2_decay += torch.sum(a1**2)

		a2 = F.relu(self.fc2(a1))
		l2_decay += torch.sum(a2**2)

		logits = self.fc3(a2)

		return logits, l2_decay

