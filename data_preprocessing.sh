#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# PARSE ARGUMENTS
# ----------------------------
DATA_DIR=""
OUT_DIR=""
SEED=123  # default seed

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
      --data_path) DATA_DIR="$2"; shift 2 ;;
      --output_dir) OUT_DIR="$2"; shift 2 ;;
      --seed) SEED="$2"; shift 2 ;;
      *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Check required args
if [[ -z "$DATA_DIR" ]] || [[ -z "$OUT_DIR" ]]; then
    echo "Usage: $0 --data_path <folder> --output_dir <folder> [--seed <int>]"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "Using DATA_DIR: $DATA_DIR"
echo "Using OUT_DIR:  $OUT_DIR"
echo "Random seed:    $SEED"

# ----------------------------
# RANDOM TRAIN/TEST SPLIT
# ----------------------------
shopt -s nullglob
FILES=( "$DATA_DIR"/*.csv )

if [ ${#FILES[@]} -lt 2 ]; then
    echo "ERROR: Need at least 2 CSV files in the data folder."
    exit 1
fi

# Seeded random selection of TRAIN file
RANDOM=$SEED
TRAIN_FILE="${FILES[$(( RANDOM % ${#FILES[@]} ))]}"
echo "Selected TRAIN file: $TRAIN_FILE"

# Remaining files = TEST
TEST_FILES=()
for f in "${FILES[@]}"; do
    if [[ "$f" != "$TRAIN_FILE" ]]; then
        TEST_FILES+=("$f")
    fi
done

echo "Test files:"
printf '%s\n' "${TEST_FILES[@]}"

# ----------------------------
# FUNCTION: extract CSV parts (first 10000 rows)
# ----------------------------
extract_parts() {
    input_file="$1"
    prefix="$2"
    index="$3"

    # Get header + first 10000 rows
    temp_file="$(mktemp)"
    head -n 10001 "$input_file" > "$temp_file"

    # Extract features (drop label + cell_id)
    awk -F',' '
        NR==1 {
            for (i=1;i<=NF;i++) {
                if ($i=="label") label_col=i;
                if ($i=="cell_id") id_col=i;
            }
        }
        {
            out=""
            for (i=1;i<=NF;i++) {
                if (i!=label_col && i!=id_col) {
                    out = out (out=="" ? $i : FS $i)
                }
            }
            print out
        }
    ' "$temp_file" > "$OUT_DIR/${prefix}_x_${index}.csv"

    # Extract labels
    awk -F',' '
        NR==1 {
            for (i=1;i<=NF;i++) { if ($i=="label") label_col=i; }
            next
        }
        { print $label_col }
    ' "$temp_file" > "$OUT_DIR/${prefix}_y_${index}.csv"

    rm "$temp_file"
}

# ----------------------------
# PROCESS TRAIN FILE
# ----------------------------
echo "Processing TRAIN file..."
extract_parts "$TRAIN_FILE" "train" 1

# ----------------------------
# PROCESS TEST FILES
# ----------------------------
echo "Processing TEST files..."
i=1
for f in "${TEST_FILES[@]}"; do
    extract_parts "$f" "test" "$i"
    i=$((i+1))
done

# ----------------------------
# ZIP outputs
# ----------------------------
echo "Creating ZIP archives..."
cd "$OUT_DIR"

zip -j train_x.zip train_x_*.csv
zip -j train_y.zip train_y_*.csv
zip -j test_x.zip  test_x_*.csv
zip -j test_y.zip  test_y_*.csv

echo "Done! Outputs written to: $OUT_DIR"
