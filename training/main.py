import numpy as np
import torch 

from . import experiments
from .execution import run_experiments 
from .config import Domain, Params
from utilities.meta import DETERMINISTIC, SEED, data_dir, out_dir, results_dir, device
from utilities.fs import make_dirs


def main():
    if DETERMINISTIC:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)

    make_dirs([data_dir, out_dir, results_dir])

    domain = Domain()
    params = Params()

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Data directory: {data_dir}, Results directory: {results_dir}")
    print(f"[INFO] Hyperparameters: {params}")
    print(f"[INFO] Domain variables: {domain}")

    experiment_builders = [
        experiments.build_experiment_1_with_replay_no_decay,
        experiments.build_experiment_2_no_replay_no_decay,
        experiments.build_experiment_3_no_replay_with_decay,
        experiments.build_experiment_4_with_replay_with_decay,
    ]

    run_experiments(experiment_builders, domain)


if __name__ == "__main__":
    main()
