#!/bin/bash
# Create the virtual environment for the YOLO26 x COCO noise experiment.
#   Usage:  source create_venv.sh    (source it so the venv stays active in your shell)
set -e

# Bootstrap uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create + activate the virtual environment (Python 3.11)
uv venv --python 3.11
source .venv/Scripts/activate     # Windows; on Linux/mac use: source .venv/bin/activate

# 1) Install the CUDA 12.6 build of PyTorch FIRST, so the next step does not pull
#    the CPU-only default wheel that ultralytics would otherwise resolve.
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2) Install the rest of the dependencies.
uv pip install -r requirements.txt

echo ""
echo "Done. Verify CUDA is available with:"
echo "    python -c \"import torch; print('cuda:', torch.cuda.is_available())\""
