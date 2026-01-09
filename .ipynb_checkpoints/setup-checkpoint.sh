#run with 
#chmod +x setup.sh
#./setup.sh


#!/bin/bash
set -e

# 1. Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3

# 2. Initialize conda for THIS script
source /workspace/miniconda3/etc/profile.d/conda.sh

# 3. Create environment
conda env create -f braided_mod.yml -a

# 4. Activate environment
conda activate braided-gpu

# 5. Install ipykernel
conda install ipykernel -y

# 6. Register Jupyter kernel
python -m ipykernel install \
  --user \
  --name braided-gpu \
  --display-name "Braided"
