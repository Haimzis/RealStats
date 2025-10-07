#!/bin/bash

# Base directory where pkls are stored
BASE_DIR="pkls/AIStats/rigid_eval"

# Declare rename rules as "old:new"
RENAMES=(
  "CNNSpotset:CNNSpot_test"
  "train2:progan"
  "768_sdxl2:768_sdv2"
  "768_sd2:768_sdv2"
  "vdqm_genimage:vqdm_genimage"
)

cd "$BASE_DIR" || exit 1

# Recursive renaming
find . -depth -type d | while read -r dir; do
  for rule in "${RENAMES[@]}"; do
    old="${rule%%:*}"
    new="${rule##*:}"
    base=$(basename "$dir")

    if [[ "$base" == "$old" ]]; then
      parent=$(dirname "$dir")
      echo "Renaming: $dir -> $parent/$new"
      mv "$dir" "$parent/$new"
      # Update variable after rename
      dir="$parent/$new"
    fi
  done
done
