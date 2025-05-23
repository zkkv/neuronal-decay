import torch
import os


DETERMINISTIC = False  # FIXME
SEED          = 43  # FIXME
SEEDS         = [43]
AVERAGE_OF    = len(SEEDS)

DATA_DIR     = os.path.expanduser("~/.cache/ml/datasets/neuronal-decay")
OUT_DIR      = "./out"
PLOTS_DIR    = f"{OUT_DIR}/plots"
RESULTS_DIR  = f"{OUT_DIR}/results"
RESULTS_FILE = f"{RESULTS_DIR}/results_{SEED}.json"
DEVICE       = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
