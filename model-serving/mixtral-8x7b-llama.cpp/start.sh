#!/bin/bash
# set -x

cat <<EOF
In order for llama-cpp-python to compile properly it needs 
to do so at run time.

EOF

python3 -m pip install --user llama-cpp-python

python3 /app/model.py
