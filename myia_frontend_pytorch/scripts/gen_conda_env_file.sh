#!/bin/bash

# NB: This script should be run from package folder containing pyproject.toml and scripts/ sub-folder.
# NB: package myia_utils must be installed.

# Install python dependencies required to run this script.
pip install poetry2conda==0.3.0

# Generate environment.yml from pyproject using poetry2conda
poetry2conda pyproject.toml environment.yml

# Generate conda environment files for CPU and GPU.
python -m myia_utils.update_env environment.yml -p cpu-extras.conda -o environment-cpu.yml
python -m myia_utils.update_env environment.yml -p gpu-extras.conda -o environment-gpu.yml
rm environment.yml
