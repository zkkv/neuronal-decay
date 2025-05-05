

import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    from torch.utils.data import Dataset


@app.class_definition
class TransformedDataset(Dataset):
    '''
    Represents a dataset with lazily-transformed value or target.
    '''

    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        (value, target) = self.dataset[index]
        if self.transform:
            value = self.transform(value)
        if self.target_transform:
            target = self.target_transform(target)
        return (value, target)


@app.class_definition
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


if __name__ == "__main__":
    app.run()
