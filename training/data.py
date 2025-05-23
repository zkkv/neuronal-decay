from torchvision import datasets, transforms
from torch.utils.data import Subset

from .config import Params
from utilities.meta import DATA_DIR
from utilities.structs import TransformedDataset


def get_datasets(data_dir, rotations):
	training_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
	test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())

	# FIXME
	# TEMPORARILY REDUCE DATASET SIZE
	training_data = Subset(training_data, range(500))
	test_data = Subset(test_data, range(500))

	train_datasets = []
	test_datasets = []
	for r in rotations:
		train_datasets.append(TransformedDataset(training_data, transform=transforms.RandomRotation(degrees=(r,r))))
		test_datasets.append(TransformedDataset(test_data, transform=transforms.RandomRotation(degrees=(r,r))))

	return training_data, test_data, train_datasets, test_datasets


# Global variables
training_data, test_data, train_datasets, test_datasets = get_datasets(DATA_DIR, Params.rotations)

print(f"[INFO] Training set size = {len(training_data)}, Test set size = {len(test_data)}")
