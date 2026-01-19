#!/usr/bin/env bash
set -euo pipefail

# Run data_preprocessing.py with the requested parameters.
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"

python "${script_dir}/data_preprocessing.py" \
  --name "data_import" \
  --output_dir "${script_dir}/out/data/data_import/preprocessing/data_preprocessing/default" \
  --data.raw "${script_dir}/out/data/data_import/data_import.data.gz" \
  --data.labels "${script_dir}/out/data/data_import/data_import.input_labels.gz"

# Create gz-suffixed symlinks where models/dgcytof expects them
repo_root="$(cd "$script_dir/.." && pwd)"
src_dir="${script_dir}/out/data/data_import/preprocessing/data_preprocessing/default"
model_out_dir="${repo_root}/models/dgcytof/out/data/data_preprocessing/default"

mkdir -p "$model_out_dir"
ln -sf "${src_dir}/data_import.matrix.gz" \
      "${model_out_dir}/data_preprocessing.csv.gz"
ln -sf "${src_dir}/data_import.true_labels.gz" \
      "${model_out_dir}/data_preprocessing_labels.txt.gz"
