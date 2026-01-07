#!/bin/bash

# Exit on any error
set -e

echo "Setting up lyra venv..."

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

VENV_NAME="lyra_env"
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment: $VENV_NAME"
    uv venv "$VENV_NAME" --python 3.12
else
    echo "Virtual environment $VENV_NAME already exists"
fi

source "$VENV_NAME/bin/activate"

# Ensure modern setuptools for building native/pyproject packages
echo "Upgrading setuptools..."
uv pip install --upgrade 'setuptools>=61'

# Install packages
echo "Installing required packages..."

uv pip install git+https://github.com/HazyResearch/flash-fft-conv.git#subdirectory=csrc/flashfftconv --no-build-isolation
uv pip install git+https://github.com/HazyResearch/flash-fft-conv.git
uv pip install 'causal_conv1d'
uv pip install 'tqdm'
uv pip install 'pyfaidx'
uv pip install 'pandas'
uv pip install 'numpy'
uv pip install 'scipy'
uv pip install 'biopython'
uv pip install 'matplotlib'
uv pip install 'pysam'
uv pip install 'seaborn'
uv pip install 'neptune-scale'
uv pip install 'scikit-learn'
uv pip install 'biotite'
uv pip install 'h5py'
uv pip install 'einops'


echo "Installation complete! To activate the virtual environment, run:"
echo "source $VENV_NAME/bin/activate"
