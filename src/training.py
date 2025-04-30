

import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    import torch
    from torchvision import datasets, transforms
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
    from torch import nn
    import torch.nn.functional as F


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Evaluation of Stability Gap in Neuronal Decay Approach""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Initial Setup""")
    return


@app.cell
def _():
    # Hyperparameters
    batch_size = 128
    rotations = [0, 160]
    learning_rate = 0.01
    n_batches_per_task = 100
    test_size = 75
    print(f"[INFO] Hyperparameters: {batch_size=}, {rotations=}, {learning_rate=}, {n_batches_per_task=}, {test_size=}")
    return batch_size, learning_rate, n_batches_per_task, rotations, test_size


@app.cell
def _():
    # Configuration
    data_dir = "~/.cache/ml/datasets/neuronal-decay"
    model_dir = "~/.cache/ml/models/neuronal-decay"
    out_dir = "./out"

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Data directory: {data_dir}, Model directory: {model_dir}, Output directory: {out_dir}")
    return data_dir, device, out_dir


@app.cell
def _(rotations, training_data):
    # Domain variables
    n_tasks = len(rotations)
    n_classes = 10
    img_n_channels = training_data[0][0].shape[0]
    img_size = training_data[0][0].shape[1]

    print(f"[INFO] Domain variables: {n_tasks=}, {n_classes=}, {img_n_channels=}, {img_size=}")
    return img_n_channels, img_size, n_classes, n_tasks


@app.cell
def _(data_dir, rotations):
    # training_data = datasets.CIFAR10(
    #     root=data_dir,
    #     train=True,
    #     download=True,
    #     transform=ToTensor(),
    # )

    # test_data = datasets.CIFAR10(
    #     root=data_dir,
    #     train=False,
    #     download=True,
    #     transform=ToTensor(),
    # )

    # training_data = datasets.FashionMNIST(
    #     root=data_dir,
    #     train=True,
    #     download=True,
    #     transform=ToTensor()
    # )

    # test_data = datasets.FashionMNIST(
    #     root=data_dir,
    #     train=False,
    #     download=True,
    #     transform=ToTensor()
    # )

    training_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())

    # TEMPORARILY REDUCE DATASET SIZE
    training_data = Subset(training_data, range(1500))
    test_data = Subset(test_data, range(1500))

    train_datasets = []
    test_datasets = []
    for r in rotations:
        train_datasets.append(TransformedDataset(training_data, transform=transforms.RandomRotation(degrees=(r,r))))
        test_datasets.append(TransformedDataset(test_data, transform=transforms.RandomRotation(degrees=(r,r))))

    print(f"[INFO] Training set size = {len(training_data)}, Test set size = {len(test_data)}")
    return test_datasets, train_datasets, training_data


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


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Model""")
    return


@app.class_definition
class Classifier(nn.Module):
    def __init__(self, img_size, img_n_channels, n_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(img_n_channels * img_size * img_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Training and Evaluation""")
    return


@app.class_definition
class Experiment:
    '''
    A single experiment involving a set of parameters and approaches.
    '''

    def __init__(self, experiment_no, model, parameters, evaluation_set, optimizer, loss_fn, use_perfect_replay, use_neuronal_decay):
        self.experiment_no = experiment_no
        self.model = model
        self.params = parameters
        self.evaluation_set = evaluation_set
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.use_perfect_replay = use_perfect_replay
        self.use_neuronal_decay = use_neuronal_decay

        self.performance = []
        self.switch_indices = []

    def set_performance_history(self, performance):
        self.performance = performance

    def set_switch_indices(self, switch_indices):
        self.switch_indices = switch_indices

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


@app.function
def compute_accuracy(model, dataset, test_size=None, batch_size=128):
    '''
    Compute accuracy (% samples classified correctly) of a classifier ([model]) on [dataset].
    '''

    mode = model.training
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    total_tested = 0
    total_correct = 0

    for X, y in dataloader:
        if test_size and total_tested >= test_size:
            break
        with torch.no_grad():
            scores = model(X)
        _, pred = torch.max(scores, 1)
        total_correct += (pred == y).sum().item()
        total_tested += len(X)
    accuracy = total_correct * 100 / total_tested

    model.train(mode=mode)

    return accuracy


@app.function
def train_and_eval(model, train_set, n_batches, batch_size, test_set, test_size, performance, optimizer, loss_fn):
    '''
    Function to train a [model] on a given [train_set],
    while evaluating after each training iteration on [test_set].
    '''
    model.train()
    iters_left = 1
    print_every_n = 25

    dataloader = CircularIterator(DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True))

    for batch_idx in range(n_batches):

        (X, y) = next(dataloader)

        # Prediction
        pred = model(X)

        # Evaluation
        accuracy = compute_accuracy(model, test_set, test_size, batch_size)
        performance.append(accuracy)
        loss = loss_fn(input=pred, target=y, reduction='mean')

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        if batch_idx % print_every_n == 0: 
            print('Training loss: {loss:04f} | Test accuracy: {prec:05.2f}% | Batch: {b_index}'
                .format(loss=loss.item(), prec=accuracy, b_index=batch_idx))


@app.cell
def _(
    batch_size,
    device,
    img_n_channels,
    img_size,
    learning_rate,
    n_batches_per_task,
    n_classes,
    rotations,
    test_datasets,
    test_size,
):
    def build_experiment_1_with_replay_no_decay():
        model = Classifier(img_size, img_n_channels, n_classes)
        model.to(device)

        params = {
            "batch_size": batch_size,
            "rotations": rotations,
            "learning_rate": learning_rate,
            "n_batches_per_task": n_batches_per_task,
            "test_size": test_size,
        }

        evaluation_set = test_datasets[0]

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        loss_fn = F.cross_entropy

        use_perfect_replay = True
        use_neuronal_decay = False

        return Experiment(1, model, params, evaluation_set, optimizer, loss_fn, use_perfect_replay, use_neuronal_decay)
    return (build_experiment_1_with_replay_no_decay,)


@app.cell
def _(
    batch_size,
    device,
    img_n_channels,
    img_size,
    learning_rate,
    n_batches_per_task,
    n_classes,
    rotations,
    test_datasets,
    test_size,
):
    def build_experiment_2_no_replay_no_decay():
        model = Classifier(img_size, img_n_channels, n_classes)
        model.to(device)

        params = {
            "batch_size": batch_size,
            "rotations": rotations,
            "learning_rate": learning_rate,
            "n_batches_per_task": n_batches_per_task,
            "test_size": test_size,
        }

        evaluation_set = test_datasets[0]

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        loss_fn = F.cross_entropy

        use_perfect_replay = False
        use_neuronal_decay = False

        return Experiment(2, model, params, evaluation_set, optimizer, loss_fn, use_perfect_replay, use_neuronal_decay)
    return (build_experiment_2_no_replay_no_decay,)


@app.cell
def _(n_tasks, train_datasets):
    def run_experiment(experiment):
        print(f" Running experiment {experiment.experiment_no} ".center(60, "~"))
        performance_history = []
        switch_indices = []

        for task_idx in range(1, n_tasks + 1):
            print(f" Training on task {task_idx} ".center(60, "="))

            if experiment.use_perfect_replay:
                dataset = ConcatDataset(train_datasets[:task_idx])
            else:
                dataset = train_datasets[task_idx - 1]

            train_and_eval(
                experiment.model,
                dataset,
                experiment.params["n_batches_per_task"],
                experiment.params["batch_size"] * task_idx,
                experiment.evaluation_set,
                experiment.params["test_size"],
                performance_history,
                experiment.optimizer,
                experiment.loss_fn,
            )
            switch_indices.append(len(performance_history))

        experiment.set_performance_history(performance_history)
        experiment.set_switch_indices(switch_indices)

        print(f"Experiment {experiment.experiment_no} done!")
    return (run_experiment,)


@app.cell
def _(
    build_experiment_1_with_replay_no_decay,
    build_experiment_2_no_replay_no_decay,
    run_experiment,
):
    experiment_1 = build_experiment_1_with_replay_no_decay()
    experiment_2 = build_experiment_2_no_replay_no_decay()

    experiments = []
    experiments.append(experiment_1)
    # experiments.append(experiment_2)

    def run_all(experiments):
        for experiment in experiments:
            run_experiment(experiment)
    run_all(experiments)
    return (experiments,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Visualization""")
    return


@app.cell
def _(experiments, plot_lines):
    plt.style.use('default')

    def plot_all(experiments):
        for experiment in experiments:
            performances = [experiment.performance]
            perf_lens = [len(l) for l in performances]
            exp_n = experiment.experiment_no

            figure = plot_lines(
                performances,
                x_axes=list(range(np.sum(perf_lens))),
                line_names=['Performance'],
                title=f"Performance on Task 1 throughout Experiment {exp_n}",
                ylabel="Test Accuracy (%) on Task 1",
                xlabel="Batch",
                figsize=(10,5),
                v_line=experiment.switch_indices[:-1],
                v_label='Task switch', ylim=(60, 100),
                save_as=f"experiment_{exp_n}.png"
            )
    plot_all(experiments)
    return


@app.cell
def _(out_dir):
    def plot_lines(list_with_lines, x_axes=None, line_names=None, colors=None, title=None,
                   title_top=None, xlabel=None, ylabel=None, ylim=None, figsize=None, list_with_errors=None, errors="shaded",
                   x_log=False, with_dots=False, linestyle='solid', h_line=None, h_label=None, h_error=None,
                   h_lines=None, h_colors=None, h_labels=None, h_errors=None,
                   v_line=None, v_label=None, save_as=None, should_show=True):
        '''Generates a figure containing multiple lines in one plot.

        :param list_with_lines: <list> of all lines to plot (with each line being a <list> as well)
        :param x_axes:          <list> containing the values for the x-axis
        :param line_names:      <list> containing the names of each line
        :param colors:          <list> containing the colors of each line
        :param title:           <str> title of plot
        :param title_top:       <str> text to appear on top of the title
        :return: f:             <figure>
        '''

        # if needed, generate default x-axis
        if x_axes == None:
            n_obs = len(list_with_lines[0])
            x_axes = list(range(n_obs))

        # if needed, generate default line-names
        if line_names == None:
            n_lines = len(list_with_lines)
            line_names = ["line " + str(line_id) for line_id in range(n_lines)]

        # make plot
        size = (12,7) if figsize is None else figsize
        f, axarr = plt.subplots(1, 1, figsize=size)

        # add error-lines / shaded areas
        if list_with_errors is not None:
            for line_id, name in enumerate(line_names):
                if errors=="shaded":
                    axarr.fill_between(x_axes, list(np.array(list_with_lines[line_id]) + np.array(list_with_errors[line_id])),
                                       list(np.array(list_with_lines[line_id]) - np.array(list_with_errors[line_id])),
                                       color=None if (colors is None) else colors[line_id], alpha=0.25)
                else:
                    axarr.plot(x_axes, list(np.array(list_with_lines[line_id]) + np.array(list_with_errors[line_id])), label=None,
                               color=None if (colors is None) else colors[line_id], linewidth=1, linestyle='dashed')
                    axarr.plot(x_axes, list(np.array(list_with_lines[line_id]) - np.array(list_with_errors[line_id])), label=None,
                               color=None if (colors is None) else colors[line_id], linewidth=1, linestyle='dashed')

        # mean lines
        for line_id, name in enumerate(line_names):
            axarr.plot(x_axes, list_with_lines[line_id], label=name,
                       color=None if (colors is None) else colors[line_id],
                       linewidth=4, marker='o' if with_dots else None, linestyle=linestyle if type(linestyle)==str else linestyle[line_id])

        # add horizontal line
        if h_line is not None:
            axarr.axhline(y=h_line, label=h_label, color="grey")
            if h_error is not None:
                if errors == "shaded":
                    axarr.fill_between([x_axes[0], x_axes[-1]],
                                       [h_line + h_error, h_line + h_error], [h_line - h_error, h_line - h_error],
                                       color="grey", alpha=0.25)
                else:
                    axarr.axhline(y=h_line + h_error, label=None, color="grey", linewidth=1, linestyle='dashed')
                    axarr.axhline(y=h_line - h_error, label=None, color="grey", linewidth=1, linestyle='dashed')

        # add horizontal lines
        if h_lines is not None:
            h_colors = colors if h_colors is None else h_colors
            for line_id, new_h_line in enumerate(h_lines):
                axarr.axhline(y=new_h_line, label=None if h_labels is None else h_labels[line_id],
                              color=None if (h_colors is None) else h_colors[line_id])
                if h_errors is not None:
                    if errors == "shaded":
                        axarr.fill_between([x_axes[0], x_axes[-1]],
                                           [new_h_line + h_errors[line_id], new_h_line+h_errors[line_id]],
                                           [new_h_line - h_errors[line_id], new_h_line - h_errors[line_id]],
                                           color=None if (h_colors is None) else h_colors[line_id], alpha=0.25)
                    else:
                        axarr.axhline(y=new_h_line+h_errors[line_id], label=None,
                                      color=None if (h_colors is None) else h_colors[line_id], linewidth=1,
                                      linestyle='dashed')
                        axarr.axhline(y=new_h_line-h_errors[line_id], label=None,
                                      color=None if (h_colors is None) else h_colors[line_id], linewidth=1,
                                      linestyle='dashed')

        # add vertical line(s)
        if v_line is not None:
            if type(v_line)==list:
                for id,new_line in enumerate(v_line):
                    axarr.axvline(x=new_line, label=v_label if id==0 else None, color='black')
            else:
                axarr.axvline(x=v_line, label=v_label, color='black')

        # finish layout
        # -set y-axis
        if ylim is not None:
            axarr.set_ylim(ylim)
        # -add axis-labels
        if xlabel is not None:
            axarr.set_xlabel(xlabel)
        if ylabel is not None:
            axarr.set_ylabel(ylabel)
        # -add title(s)
        if title is not None:
            axarr.set_title(title)
        if title_top is not None:
            f.suptitle(title_top)
        # -add legend
        if line_names is not None:
            axarr.legend()
        # -set x-axis to log-scale
        if x_log:
            axarr.set_xscale('log')
        if save_as is not None:
            full_path="{}/{}".format(out_dir, save_as)
            plt.savefig(full_path)
        if should_show:
            plt.show()


        # return the figure
        return f
    return (plot_lines,)


@app.cell(hide_code=True)
def _():
    ## Utilities
    return


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
