#!/usr/bin/env bash
# Run the training program on multiple seeds and generate plots and metrics.

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
seeds=(
  290353
  105883
  942000
  2958
  876171
  42775
  948129
  446869
  9782
  153441
  72745
  777330
  372621
  94467
  812518
)

for seed in "${seeds[@]}"; do
  poetry run python -m training.main -s $seed
done

poetry run python -m analysis.main "${seeds[@]}"
