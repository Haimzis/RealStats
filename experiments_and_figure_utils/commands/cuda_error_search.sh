#!/bin/bash

# Define the pattern to match CUDA errors
CUDA_ERROR_PATTERN="CUDA"

# Search through each logs.txt file
for logfile in logs/run*/logs.txt; do
    if grep -qE "$CUDA_ERROR_PATTERN" "$logfile"; then
        # Extract and print the run directory name (e.g., run1)
        dirname=$(basename "$(dirname "$logfile")")
        echo "$dirname"
    fi
done
