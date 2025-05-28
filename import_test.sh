#!/usr/bin/env bash
# Confirm that all required dependencies are found.

# Poetry and PyTorch don't like each other, so their friendship requires some assistance.
# In general, poetry can install torch/torchvision successfully, but for some reason
# there might be issues with torch not seeing the three dynamic libraries (shared objects) below,
# even though they're actually installed by poetry within the virtual environment. So, we export them.
# # Feel free to modify the path(s) or delete this part if it works for you out-of-the-box.
POETRY_VENV_PATH=$(poetry env info --path)
POETRY_VENV_PACKAGES="$POETRY_VENV_PATH/lib/python3.12/site-packages"

export LD_LIBRARY_PATH="$POETRY_VENV_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$POETRY_VENV_PACKAGES/nvidia/nccl/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$POETRY_VENV_PACKAGES/cusparselt/lib:$LD_LIBRARY_PATH"

# Actual script
poetry run python utilities/import_test.py
