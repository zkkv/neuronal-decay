

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
    import json
    from utilities import TransformedDataset, CircularIterator, average_inhomogeneous

    DETERMINISTIC = False  # FIXME
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
    rotations = [0, 160] # FIXME
    learning_rate = 0.01
    n_batches_per_task = 1 # FIXME
    test_size = 512
    average_of = 2  # FIXME
    decay_lambda = 1e-7

    print(f"[INFO] Hyperparameters: {batch_size=}, {rotations=}, {learning_rate=}, {n_batches_per_task=}, {test_size=}, {average_of=}")
    return (
        average_of,
        batch_size,
        decay_lambda,
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
    results_file = f"{out_dir}/results/results.json"

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Data directory: {data_dir}, Model directory: {model_dir}, Output directory: {out_dir}")
    return data_dir, device, results_file


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
    # training_data = Subset(training_data, range(2000))
    # test_data = Subset(test_data, range(2000))
    # training_data = Subset(training_data, range(5000))
    # test_data = Subset(test_data, range(5000))

    train_datasets = []
    test_datasets = []
    for r in rotations:
        train_datasets.append(TransformedDataset(training_data, transform=transforms.RandomRotation(degrees=(r,r))))
        test_datasets.append(TransformedDataset(test_data, transform=transforms.RandomRotation(degrees=(r,r))))

    print(f"[INFO] Training set size = {len(training_data)}, Test set size = {len(test_data)}")
    return test_data, test_datasets, train_datasets, training_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Model""")
    return


@app.class_definition
class Classifier(nn.Module):
    def __init__(self, img_size, img_n_channels, n_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(img_n_channels * img_size * img_size, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, n_classes)


    def forward(self, x):
        x = self.flatten(x)
        l2_decay = 0

        a1 = F.relu(self.fc1(x))
        l2_decay += torch.sum(a1**2)

        a2 = F.relu(self.fc2(a1))
        l2_decay += torch.sum(a2**2)

        logits = self.fc3(a2)

        return logits, l2_decay


@app.cell
def _(img_n_channels, img_size, n_classes):
    def get_model():
        return Classifier(img_size, img_n_channels, n_classes)
    return (get_model,)


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

    def __init__(self, experiment_no, model, parameters, evaluation_sets, optimizer, loss_fn, use_perfect_replay):
        self.experiment_no = experiment_no
        self.model = model
        self.params = parameters
        self.evaluation_sets = evaluation_sets
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.use_perfect_replay = use_perfect_replay

        self.performances = []
        self.switch_indices = []

    def set_performance_history(self, performances):
        self.performances = performances

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

    def __init__(self, experiment_no, performances, switch_indices, stds=None, parameters=None, use_perfect_replay=None):
        self.experiment_no = experiment_no
        self.performances = performances
        self.switch_indices = switch_indices
        self.stds = stds
        self.params = parameters
        self.use_perfect_replay = use_perfect_replay

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
            scores, _ = model(X)
        _, pred = torch.max(scores, 1)
        total_correct += (pred == y).sum().item()
        total_tested += len(X)
    accuracy = total_correct * 100 / total_tested

    model.train(mode=mode)

    return accuracy


@app.function
def train_and_eval(model, train_set, n_batches, batch_size, task_idx, test_sets, test_size, performance, optimizer, loss_fn, decay_lambda):
    '''
    Function to train a [model] on a given [train_set],
    while evaluating after each training iteration on [test_sets].
    '''
    model.train()
    iters_left = 1
    print_every_n = 25

    dataloader = CircularIterator(DataLoader(train_set, batch_size=batch_size, shuffle=True))

    for batch_idx in range(n_batches):

        (X, y) = next(dataloader)

        # Prediction
        pred, decay = model(X)

        # Evaluation
        loss = loss_fn(pred, y) + decay_lambda * decay

        for test_idx, test_set in enumerate(test_sets):
            if test_idx >= task_idx:
                break
            accuracy = compute_accuracy(model, test_set, test_size, batch_size)
            performance[test_idx].append(accuracy)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        if batch_idx % print_every_n == 0: 
            print('Training loss: {loss:04f} | Test accuracy: {prec:05.2f}% | Batch: {b_index}'
                .format(loss=loss.item(), prec=accuracy, b_index=batch_idx))


@app.function
def ce_loss(prediction, y, reduction='mean'):
    return F.cross_entropy(input=prediction, target=y, reduction=reduction)


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Execution""")
    return


@app.cell
def _(
    batch_size,
    device,
    get_model,
    learning_rate,
    n_batches_per_task,
    rotations,
    test_datasets,
    test_size,
):
    def build_experiment_1_with_replay_no_decay():
        model = get_model()
        model.to(device)

        params = {
            "batch_size": batch_size,
            "rotations": rotations,
            "learning_rate": learning_rate,
            "n_batches_per_task": n_batches_per_task,
            "test_size": test_size,
            "decay_lambda": 0,
        }

        evaluation_sets = test_datasets

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        loss_fn = ce_loss

        use_perfect_replay = True

        return Experiment(1, model, params, evaluation_sets, optimizer, loss_fn, use_perfect_replay)
    return (build_experiment_1_with_replay_no_decay,)


@app.cell
def _(
    batch_size,
    device,
    get_model,
    learning_rate,
    n_batches_per_task,
    rotations,
    test_datasets,
    test_size,
):
    def build_experiment_2_no_replay_no_decay():
        model = get_model()
        model.to(device)

        params = {
            "batch_size": batch_size,
            "rotations": rotations,
            "learning_rate": learning_rate,
            "n_batches_per_task": n_batches_per_task,
            "test_size": test_size,
            "decay_lambda": 0,
        }

        evaluation_sets = test_datasets

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        loss_fn = ce_loss

        use_perfect_replay = False

        return Experiment(2, model, params, evaluation_sets, optimizer, loss_fn, use_perfect_replay)
    return (build_experiment_2_no_replay_no_decay,)


@app.cell
def _(
    batch_size,
    decay_lambda,
    device,
    get_model,
    learning_rate,
    n_batches_per_task,
    rotations,
    test_datasets,
    test_size,
):
    def build_experiment_3_no_replay_with_decay():
        model = get_model()
        model.to(device)

        params = {
            "batch_size": batch_size,
            "rotations": rotations,
            "learning_rate": learning_rate,
            "n_batches_per_task": n_batches_per_task,
            "test_size": test_size,
            "decay_lambda": decay_lambda,
        }

        evaluation_sets = test_datasets

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        loss_fn = ce_loss

        use_perfect_replay = False

        return Experiment(3, model, params, evaluation_sets, optimizer, loss_fn, use_perfect_replay)
    return (build_experiment_3_no_replay_with_decay,)


@app.cell
def _(
    batch_size,
    decay_lambda,
    device,
    get_model,
    learning_rate,
    n_batches_per_task,
    rotations,
    test_datasets,
    test_size,
):
    def build_experiment_4_with_replay_with_decay():
        model = get_model()
        model.to(device)

        params = {
            "batch_size": batch_size,
            "rotations": rotations,
            "learning_rate": learning_rate,
            "n_batches_per_task": n_batches_per_task,
            "test_size": test_size,
            "decay_lambda": decay_lambda,
        }

        evaluation_sets = test_datasets

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        loss_fn = ce_loss

        use_perfect_replay = True

        return Experiment(4, model, params, evaluation_sets, optimizer, loss_fn, use_perfect_replay)
    return (build_experiment_4_with_replay_with_decay,)


@app.cell
def _(average_of, n_batches_per_task, n_tasks, train_datasets):
    def run_experiment(experiment, run):
        print(f" Running experiment {experiment.experiment_no} (run {run}/{average_of}) ".center(60, "~"))
        performance_history = [[] for _ in range(n_tasks)]
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
                task_idx,
                experiment.evaluation_sets,
                experiment.params["test_size"],
                performance_history,
                experiment.optimizer,
                experiment.loss_fn,
                experiment.params["decay_lambda"],
            )
            switch_indices.append(task_idx * n_batches_per_task)

        experiment.set_performance_history(performance_history)
        experiment.set_switch_indices(switch_indices)
    return (run_experiment,)


@app.cell
def _(average_of, results_file, run_experiment, save_results_to_file):
    def run_experiments(experiment_builders, persist_results=True):
        results = []
        for eb in experiment_builders:
            runs = []
            for r in range(1, average_of + 1):
                e = eb()
                run_experiment(e, r)
                runs.append(e)

            avg_performances, stds = average_inhomogeneous([r.performances for r in runs])
            res = ExperimentResult(
                runs[0].experiment_no,
                avg_performances,
                runs[0].switch_indices,
                stds=stds,
                parameters=runs[0].params,
                use_perfect_replay=runs[0].use_perfect_replay
            )
            results.append(res)
            print(f"Experiment {res.experiment_no} done!")

        if persist_results:
            save_results_to_file(results, results_file)

        print("ALL EXPERIMENTS DONE!")
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
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Utilities""")
    return


@app.cell
def _(
    average_of,
    img_n_channels,
    img_size,
    n_classes,
    n_tasks,
    test_data,
    training_data,
):
    def save_results_to_file(results, results_file, should_log=True):
        mapped = {}

        for res in results:
            # Note: We save most values just for logging purposes.
            #       Not everything is used for evaluation/visualization.
            domain_vars = {
                "n_tasks": n_tasks,
                "n_classes": n_classes,
                "img_n_channels": img_n_channels,
                "img_size": img_size,
                "len(training_data)": len(training_data),
                "len(test_data)": len(test_data),
            }

            mapped_experiment = {
                "experiment_no": res.experiment_no,
                "performances": res.performances,
                "switch_indices": res.switch_indices,
                "stds": res.stds,
                "parameters": res.params,
                "use_perfect_replay": res.use_perfect_replay,
                "average_of": average_of,
                "domain_variables": domain_vars,
            }
            mapped[f"{res.experiment_no}"] = mapped_experiment

        with open(results_file, 'w') as f:
            if should_log:
                print(f"[INFO] Saving results to {results_file}")
            json.dump(mapped, f, indent=2)
    return (save_results_to_file,)


if __name__ == "__main__":
    app.run()
