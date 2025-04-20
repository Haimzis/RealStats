#!/bin/bash

# Check for config file argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_config_json>"
    exit 1
fi

CONFIG_FILE=$1

# Base directory for logs
LOGS_BASE_DIR="logs"

# Load configurations from the provided JSON file
CONFIGS=$(cat "$CONFIG_FILE")
CONFIGS_LENGTH=$(echo "$CONFIGS" | jq '. | length')

# Loop through configurations
for i in $(seq 0 $((CONFIGS_LENGTH - 1))); do
    CONFIG=$(echo "$CONFIGS" | jq -r ".[$i]")

    # Generate timestamped logs directory
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOGS_DIR="$LOGS_BASE_DIR/run_$TIMESTAMP"
    mkdir -p "$LOGS_DIR"

    # Redirect all output to logs.txt
    {
        echo "Starting run $((i + 1))..."
        echo "Command: python executor.py $CONFIG"

        python executor.py \
            --test_type multiple_patches \
            --batch_size 64 \
            --sample_size 256 \
            --threshold 0.05 \
            --save_histograms 1 \
            --ensemble_test manual-stouffer \
            --save_independence_heatmaps 1 \
            --uniform_sanity_check 0 \
            --output_dir "$LOGS_DIR" \
            --pkls_dir rigid_pkls \
            --num_samples_per_class -1 \
            --num_data_workers 4 \
            --max_workers 1 \
            --seed 42 \
            --wavelet_levels 0 \
            --cdf_bins 2000 \
            --n_trials 75 \
            --uniform_p_threshold 0.05 \
            --calibration_auc_threshold 0.5 \
            --ks_pvalue_abs_threshold 0.4 \
            --minimal_p_threshold 0.1 \
            --run_id "run_$TIMESTAMP" \
            --experiment_id "R minp - Experiments I" \
            --gpu 3 \
            $CONFIG

        echo "Run $((i + 1)) complete."
    } &> "$LOGS_DIR/logs.txt"
done
