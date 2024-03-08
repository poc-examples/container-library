#!/bin/bash
set -x

huggingface-cli download 'mistralai/Mixtral-8x7B-Instruct-v0.1' --cache-dir /ai-models

python3 /app/model.py
