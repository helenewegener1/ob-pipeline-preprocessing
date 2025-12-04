#!/usr/bin/env bash
set -euo pipefail

# Run data_preprocessing.py with the requested parameters.
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"

python "${script_dir}/data_preprocessing.py" \
  --name "data_import" \
  --output_dir "${script_dir}/out/data/data_import/preprocessing/data_preprocessing/default" \
  --data.raw "${script_dir}/out/data/data_import/data_import.data.gz" \
  --data.labels "${script_dir}/out/data/data_import/data_import.input_labels.gz"
