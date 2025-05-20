import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import json
    import os
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


@app.function
def average_inhomogeneous(xsss, compute_std=True):
    """
    Average lists of the same shape along the first dimension.

    Each element of xsss is a list xss such that len(xss[i]) is not necessarily equal to len(xss[j]) for i != j. However,
    len(xsss[i]) must always be equal to len(xsss[j]) for any i, j.

    This function returns a list of averages as the first element and 
    a list of standard deviations as the second element, both of the same shape as xsss[i] for any i.

    Example:
    a = [[1, 2, 3], [10, 20, 30, 40], [100]]
    b = [[4, 5, 6], [50, 60, 70, 80], [600]]

    >>> average_inhomogeneous([a, b])[0]
    [[2.5, 3.5, 4.5], [30, 40, 50, 60], [350]]
    """
    n_sublists = len(xsss[0])
    avgs = []
    stds = []

    for i in range(n_sublists):
        group = [xss[i] for xss in xsss]
        stacked = np.stack(group, axis=0)
        avg = stacked.mean(axis=0).tolist()
        avgs.append(avg)

        if compute_std:
            std = stacked.std(axis=0, ddof=1).tolist()
            stds.append(std)

    if compute_std:
        return avgs, stds
    return avgs, None


@app.function
def make_dirs(dirs):
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


if __name__ == "__main__":
    app.run()
