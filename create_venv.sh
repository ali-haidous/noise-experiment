#!/bin/bash

# Bootstrap uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment with Python 3.11 using uv
uv venv --python 3.11

# Activate the virtual environment (Windows path)
source .venv/Scripts/activate

# Install requirements using uv (much faster than pip)
uv pip install -r requirements.txt

# Install PyTorch with CUDA 12.6 support using uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126