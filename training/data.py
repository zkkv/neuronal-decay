from torchvision import datasets, transforms
from torch.utils.data import Subset

from utilities.structs import TransformedDataset


def get_datasets(data_dir, rotations, limit=None):
	training_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
	test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())

	if limit is not None:
		training_data = Subset(training_data, range(limit))
		test_data = Subset(test_data, range(limit))

	train_datasets = []
	test_datasets = []
	for r in rotations:
		train_datasets.append(TransformedDataset(training_data, transform=transforms.RandomRotation(degrees=(r,r))))
		test_datasets.append(TransformedDataset(test_data, transform=transforms.RandomRotation(degrees=(r,r))))

	return train_datasets, test_datasets
