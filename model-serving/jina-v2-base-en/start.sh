#!/bin/bash
set -x

huggingface-cli download 'jinaai/jina-embeddings-v2-base-en' --cache-dir /ai-models

python3 /app/model.py
