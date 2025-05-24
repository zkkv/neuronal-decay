import torch
import os


DATA_DIR     = os.path.expanduser("~/.cache/ml/datasets/neuronal-decay")
OUT_DIR      = "./out"
PLOTS_DIR    = f"{OUT_DIR}/plots"
RESULTS_DIR  = f"{OUT_DIR}/results"
LOG_DIR      = f"{OUT_DIR}/logs"
DEVICE       = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
