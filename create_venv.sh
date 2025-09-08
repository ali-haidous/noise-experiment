#!/bin/bash
python -m venv venv
source venv/Scripts/activate venv
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126