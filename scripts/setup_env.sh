#!/usr/bin/env bash
# Sets up the Python environment on HPC or local machine.
# Run once after cloning/copying the project.
# Usage: bash scripts/setup_env.sh

set -e

echo "Setting up environment..."

# check if conda is available
if command -v conda &> /dev/null; then
    echo "conda found — creating conda environment"
    conda create --name t2i python=3.10 -y
    conda activate t2i
else
    echo "conda not found — creating venv"
    python3 -m venv t2i
    source t2i/bin/activate
fi

pip install --upgrade pip
pip install -r requirements.txt

echo "Done. Environment ready."
echo "Activate with: source t2i/bin/activate"
