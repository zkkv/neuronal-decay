

import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torchvision import datasets, transforms
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
    from torch import nn
    import torch.nn.functional as F

    DETERMINISTIC = True
    SEED = 42

    if DETERMINISTIC:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)


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
    batch_size = 512
    rotations = [0, 30, 160]
    learning_rate = 0.01
    n_batches_per_task = 100 # FIXME
    test_size = 512
    average_of = 5  # FIXME

    print(f"[INFO] Hyperparameters: {batch_size=}, {rotations=}, {learning_rate=}, {n_batches_per_task=}, {test_size=}, {average_of=}")
    return (
        average_of,
        batch_size,
        learning_rate,
        n_batches_per_task,
        rotations,
        test_size,
    )


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

    # FIXME
    # TEMPORARILY REDUCE DATASET SIZE
    training_data = Subset(training_data, range(500))
    test_data = Subset(test_data, range(500))

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
        self.fc1 = nn.Linear(img_n_channels * img_size * img_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, n_classes)

        self.activations = {}


    def forward(self, x):
        x = self.flatten(x)

        a1 = F.relu(self.fc1(x))
        self.activations[1] = a1

        a2 = F.relu(self.fc2(a1))
        self.activations[2] = a2

        logits = self.fc3(a2)
        return logits


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Training and Evaluation""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Setup""")
    return


@app.class_definition
class Experiment:
    '''
    A single experiment involving a set of parameters and approaches.
    '''

    def __init__(self, experiment_no, model, parameters, evaluation_set, optimizer, loss_fn, use_perfect_replay):
        self.experiment_no = experiment_no
        self.model = model
        self.params = parameters
        self.evaluation_set = evaluation_set
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.use_perfect_replay = use_perfect_replay

        self.performance = []
        self.switch_indices = []

    def set_performance_history(self, performance):
        self.performance = performance

    def set_switch_indices(self, switch_indices):
        self.switch_indices = switch_indices

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


@app.class_definition
class ExperimentResult:
    '''
    A structure wrapping experiment results.
    '''

    def __init__(self, experiment_no, performance, switch_indices):
        self.experiment_no = experiment_no
        self.performance = performance
        self.switch_indices = switch_indices


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
def train_and_eval(model, train_set, n_batches, batch_size, test_set, test_size, performance, optimizer, loss_fn, decay_lambda):
    '''
    Function to train a [model] on a given [train_set],
    while evaluating after each training iteration on [test_set].
    '''
    model.train()
    iters_left = 1
    print_every_n = 25

    dataloader = CircularIterator(DataLoader(train_set, batch_size=batch_size, shuffle=True))

    for batch_idx in range(n_batches):

        (X, y) = next(dataloader)

        # Prediction
        pred = model(X)

        # Evaluation
        accuracy = compute_accuracy(model, test_set, test_size, batch_size)
        performance.append(accuracy)
        loss = loss_fn(pred, y, activations=model.activations, decay_lambda=decay_lambda)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        if batch_idx % print_every_n == 0: 
            print('Training loss: {loss:04f} | Test accuracy: {prec:05.2f}% | Batch: {b_index}'
                .format(loss=loss.item(), prec=accuracy, b_index=batch_idx))


@app.function
def loss_baseline(prediction, y, reduction='mean', activations=None, decay_lambda=0):
    return F.cross_entropy(input=prediction, target=y, reduction=reduction)


@app.function
def neuronal_decay_l2(activations):
    l2_decay = 0.0
    for layer in activations:
        l2_decay += torch.sum(activations[layer]**2)
    return l2_decay


@app.function
def loss_with_l2(prediction, y, reduction='mean', activations=None, decay_lambda=0):
    return loss_baseline(prediction, y, reduction) + decay_lambda * neuronal_decay_l2(activations)


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Execution""")
    return


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
            "decay_lambda": 0,
        }

        evaluation_set = test_datasets[0]

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        loss_fn = loss_baseline

        use_perfect_replay = True

        return Experiment(1, model, params, evaluation_set, optimizer, loss_fn, use_perfect_replay)
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
            "decay_lambda": 0,
        }

        evaluation_set = test_datasets[0]

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        loss_fn = loss_baseline

        use_perfect_replay = False

        return Experiment(2, model, params, evaluation_set, optimizer, loss_fn, use_perfect_replay)
    return (build_experiment_2_no_replay_no_decay,)


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
    def build_experiment_3_no_replay_with_decay():
        model = Classifier(img_size, img_n_channels, n_classes)
        model.to(device)

        params = {
            "batch_size": batch_size,
            "rotations": rotations,
            "learning_rate": learning_rate,
            "n_batches_per_task": n_batches_per_task,
            "test_size": test_size,
            "decay_lambda": 4e-3,
        }

        evaluation_set = test_datasets[0]

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        loss_fn = loss_with_l2

        use_perfect_replay = False

        return Experiment(3, model, params, evaluation_set, optimizer, loss_fn, use_perfect_replay)
    return (build_experiment_3_no_replay_with_decay,)


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
    def build_experiment_4_with_replay_with_decay():
        model = Classifier(img_size, img_n_channels, n_classes)
        model.to(device)

        params = {
            "batch_size": batch_size,
            "rotations": rotations,
            "learning_rate": learning_rate,
            "n_batches_per_task": n_batches_per_task,
            "test_size": test_size,
            "decay_lambda": 4e-3,
        }

        evaluation_set = test_datasets[0]

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        loss_fn = loss_with_l2

        use_perfect_replay = True

        return Experiment(4, model, params, evaluation_set, optimizer, loss_fn, use_perfect_replay)
    return (build_experiment_4_with_replay_with_decay,)


@app.cell
def _(n_tasks, train_datasets):
    def run_experiment(experiment, run):
        print(f" Running experiment {experiment.experiment_no} (run {run}) ".center(60, "~"))
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
                experiment.params["decay_lambda"],
            )
            switch_indices.append(len(performance_history))

        experiment.set_performance_history(performance_history)
        experiment.set_switch_indices(switch_indices)
    return (run_experiment,)


@app.cell
def _(average_of, run_experiment, save_result_to_csv):
    def run_experiments(experiment_builders, persist_results=True):
        results = []
        for eb in experiment_builders:
            runs = []
            for r in range(1, average_of + 1):
                e = eb()
                run_experiment(e, r)
                runs.append(e)

            avg_performance = np.mean([r.performance for r in runs], axis=0)
            res = ExperimentResult(runs[0].experiment_no, avg_performance, runs[0].switch_indices)
            results.append(res)
            print(f"Experiment {res.experiment_no} done!")

            if persist_results:
                save_result_to_csv(res)

        return results
    return (run_experiments,)


@app.cell
def _(
    build_experiment_1_with_replay_no_decay,
    build_experiment_2_no_replay_no_decay,
    build_experiment_3_no_replay_with_decay,
    build_experiment_4_with_replay_with_decay,
    run_experiments,
):
    experiment_builders = [
        build_experiment_1_with_replay_no_decay,
        build_experiment_2_no_replay_no_decay,
        build_experiment_3_no_replay_with_decay,
        build_experiment_4_with_replay_with_decay,
    ]

    results = run_experiments(experiment_builders)
    return (results,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Visualization""")
    return


@app.cell
def _(plot_lines, results):
    plt.style.use('default')

    def plot_individually(results):
        for result in results:
            performances = [result.performance]
            exp_n = result.experiment_no

            figure = plot_lines(
                performances,
                line_names=['Performance'],
                title=f"Performance on Task 1 throughout Experiment {exp_n}",
                ylabel="Test Accuracy (%) on Task 1",
                xlabel="Batch",
                figsize=(10,5),
                v_line=result.switch_indices[:-1],
                v_label='Task switch', ylim=(60, 100),
                save_as=f"plots/experiment_{exp_n}.svg"
            )


    def plot_all(results):
        performances = [e.performance for e in results]
        exp_ns = [e.experiment_no for e in results]

        figure = plot_lines(
            performances,
            line_names=[f'Experiment {exp_n}' for exp_n in exp_ns],
            title=f"Performance on Task 1 throughout Experiment(s): {exp_ns}",
            ylabel="Test Accuracy (%) on Task 1",
            xlabel="Batch",
            figsize=(10,5),
            v_line=results[0].switch_indices[:-1],
            v_label='Task switch', ylim=(70, 100),
            save_as=f"plots/experiment_{exp_ns}.svg"
        )

    plot_all(results)
    return


@app.cell(hide_code=True)
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
            plt.savefig(full_path, bbox_inches='tight')
        if should_show:
            plt.show()


        # return the figure
        return f
    return (plot_lines,)


@app.cell(hide_code=True)
def _():
    ## Metrics
    return


@app.function
def accuracy_at_task_switches(performance, switch_indices):
    res = []
    for i in switch_indices:
        res.append(performance[i - 1])
    return res


@app.function
def gap_depths(performance, switch_indices):
    accs = accuracy_at_task_switches(performance, switch_indices)
    accs.pop()

    res = []
    for i, acc in enumerate(accs):
        start, end = switch_indices[i], switch_indices[i + 1]
        min_per_task = min(performance[start:end])
        res.append(acc - min_per_task)
    return res


@app.cell
def _(results):
    print("Accuracy at task switches:")
    for e in results:
        print(accuracy_at_task_switches(e.performance, e.switch_indices))

    print("\nGap depths:")
    for e in results:
        print(gap_depths(e.performance, e.switch_indices))
    return


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


@app.cell
def _(out_dir):
    def save_result_to_csv(result):
        name = f"perf_{result.experiment_no}.csv"
        np.savetxt(f"{out_dir}/csv/{name}", result.performance, delimiter=",", fmt='%s')
    
        name = f"switch_indices_{result.experiment_no}.csv"
        np.savetxt(f"{out_dir}/csv/{name}", result.switch_indices, delimiter=",", fmt='%s')
    return (save_result_to_csv,)


if __name__ == "__main__":
    app.run()
