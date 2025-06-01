from torchvision import datasets, transforms
from torch.utils.data import Subset

from utilities.structs import TransformedDataset


def get_datasets(data_dir, rotations):
	training_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
	test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())

	train_datasets = []
	test_datasets = []
	for r in rotations:
		train_datasets.append(TransformedDataset(training_data, transform=transforms.RandomRotation(degrees=(r,r))))
		test_datasets.append(TransformedDataset(test_data, transform=transforms.RandomRotation(degrees=(r,r))))

	return train_datasets, test_datasets
