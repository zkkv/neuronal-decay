#!/usr/bin/env bash
# Run the training program on multiple seeds for profiling.

# Poetry and PyTorch don't like each other, so their friendship requires some assistance.
# In general, poetry can install torch/torchvision successfully, but for some reason
# there might be issues with torch not seeing the three dynamic libraries (shared objects) below,
# even though they're actually installed by poetry within the virtual environment. So, we export them.
# Feel free to modify the path(s) or delete this part if it works for you out-of-the-box.
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
  157186
  622015
  165838
  502328
  708216
  127911
  547089
  916014
  984231
  86496
  266748
  972119
  907599
  665579
  70875
  558521
  96013
  299954
  982820
  191954
  262342
  336966
  590552
  684652
  223248
  1559
  149701
  533710
  619838
  740084
  344189
  359699
  509966
  441554
  953448
  708295
  222791
  351805
  901197
  931503
  273005
  271616
  56026
  588746
  415326
)

for seed in "${seeds[@]}"; do
  poetry run python -m training.main -s $seed -p -r
done
