import numpy as np
from dataclasses import asdict 


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


def str_struct(ref):
    items = asdict(ref)
    return ', '.join(f"{k}={v}" for k, v in items.items())
