#!/usr/bin/env bash

# Change To Root Directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ..

module load release/25.06  GCCcore/13.3.0  Python/3.12.3
python3 -m venv --system-site-package ./hpc/venvs/venv
source ./hpc/venvs/venv/bin/activate
pip install -r requirements.txt