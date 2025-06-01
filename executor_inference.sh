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

        python executor_inference.py \
            --ensemble_test "manual-stouffer" \
            --batch_size 32 \
            --output_dir "$LOGS_DIR" \
            --num_data_workers 3 \
            --max_workers 4 \
            --experiment_id "R stouffer no_patch - Experiments II - Ours" \
            --gpu "0,1,2,3" \
            $CONFIG

        echo "Run $((i + 1)) complete."
    } &> "$LOGS_DIR/logs.txt"
done
