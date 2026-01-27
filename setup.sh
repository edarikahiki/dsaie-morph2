#----------------------------------------------------------------------------------------
# 1. IF CONDA & ENVIRONMENT ARE NOT INSTALLED IN NEW MACHINE, RUN THIS
#----------------------------------------------------------------------------------------

# how to run
# cd /workspace
# chmod +x setup.sh
# ./setup.sh


#!/bin/bash
set -e

# Always run relative to /workspace
cd /workspace

# ----------------------------
# Config
# ----------------------------
CONDA_DIR="/workspace/miniconda3"
ENV_NAME="braided-gpu"
ENV_YML="/workspace/braided_mod.yml"

# ----------------------------
# 1. Download Miniconda (only if missing)
# ----------------------------
if [ ! -d "$CONDA_DIR" ]; then
    wget -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3.sh -b -p "$CONDA_DIR"
    rm Miniconda3.sh
fi

# ----------------------------
# 2. Initialize conda for THIS script
# ----------------------------
source "$CONDA_DIR/etc/profile.d/conda.sh"

# ----------------------------
# 3. Create env (only if missing)
# ----------------------------
if ! conda env list | grep -q "$ENV_NAME"; then
    conda env create -f "$ENV_YML"
fi

# ----------------------------
# 4. Activate environment
# ----------------------------
conda activate "$ENV_NAME"

# ----------------------------
# 5. Install ipykernel
# ----------------------------
conda install -y ipykernel

# ----------------------------
# 6. Register Jupyter kernel
# ----------------------------
python -m ipykernel install \
  --user \
  --name "$ENV_NAME" \
  --display-name "Braided"


#-------------------------------------------------------------------------------------------------------
#2. IF CONDA & ENVIRONMENT ARE INSTALLED, RUN THIS TO ACTIVATE THE CORRECT KERNEL
#-------------------------------------------------------------------------------------------------------

# check if conda is installed
ls /workspace/miniconda3/bin/conda \
&& source /workspace/miniconda3/etc/profile.d/conda.sh \
&& conda --version

# check if conda is installed
python -m ipykernel install \
  --user \
  --name braided-gpu \
  --display-name "Braided (conda)"
