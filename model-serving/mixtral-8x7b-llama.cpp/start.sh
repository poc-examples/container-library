#!/bin/bash
# set -x

nvcc --version

nvidia-smi

cat <<EOF
In order for llama-cpp-python to compile properly it needs 
to do so at run time.

EOF

export USE_CUDA=1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python3 -m pip install --user llama-cpp-python

touch /var/log/nvidia-mps && nvidia-cuda-mps-control -d && python3 /app/main.py
