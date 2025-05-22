import torch
import os


DETERMINISTIC = False  # FIXME
SEED          = 43  # FIXME
SEEDS         = [43]
average_of    = len(SEEDS)

data_dir     = os.path.expanduser("~/.cache/ml/datasets/neuronal-decay")
out_dir      = "./out"
plots_dir    = f"{out_dir}/plots"
results_dir  = f"{out_dir}/results"
results_file = f"{results_dir}/results_{SEED}.json"
device       = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
