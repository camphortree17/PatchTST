#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /home/hmq/miniconda3/etc/profile.d/conda.sh
conda activate PatchTST

export CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/train.yaml}"

python -u "${SCRIPT_DIR}/train_itransformer.py"
