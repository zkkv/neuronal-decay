

import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import random
    import torch
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader
    return DataLoader, ToTensor, datasets, plt, random


@app.cell
def _():
    # Hyperparameters
    batch_size = 64
    return (batch_size,)


@app.cell
def _(DataLoader, ToTensor, batch_size, datasets):
    data_dir = "data"

    training_data = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return (training_data,)


@app.cell
def _(plt, random, training_data):
    def plot_examples(data_tensor):
        h, w = 4, 4
        fig, axes = plt.subplots(h, w, figsize=(6, 6))
        indices = random.sample(range(len(data_tensor)), h * w)

        labels_map = {
            0: "Airplane",
            1: "Car",
            2: "Bird",
            3: "Cat",
            4: "Deer",
            5: "Dog",
            6: "Frog",
            7: "Horse",
            8: "Ship",
            9: "Truck",
        }
    
        for ax, idx in zip(axes.flatten(), indices):
            img, label = data_tensor[idx]
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(f"Label: {labels_map[label]}\nIndex: {idx}")
            ax.axis('off')
    
        plt.tight_layout()
        plt.show()
    plot_examples(training_data)
    return


if __name__ == "__main__":
    app.run()
