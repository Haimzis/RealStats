#!/bin/bash

# Define dataset paths
DATA_DIR_REAL="data/CelebaHQMaskDataset/train/images_faces"
DATA_DIR_FAKE_REAL="data/CelebaHQMaskDataset/test/images_faces"
DATA_DIR_FAKE="data/stable-diffusion-face-dataset/1024/both_faces"

# Base directory for logs
LOGS_BASE_DIR="logs"

# Load configurations from JSON file
CONFIGS=$(cat configs.json)
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
            --batch_size 256 \
            --data_dir_real "$DATA_DIR_REAL" \
            --data_dir_fake_real "$DATA_DIR_FAKE_REAL" \
            --data_dir_fake "$DATA_DIR_FAKE" \
            --output_dir "$LOGS_DIR" \
            --pkls_dir "/data/users/haimzis/pkls" \
            --num_samples_per_class 2957 \
            --num_data_workers 2 \
            --max_wave_level 4 \
            --max_workers 32 \
            --seed 42 \
            --run_id "run_$TIMESTAMP" \
            $CONFIG

        echo "Run $((i + 1)) complete."
    } &> "$LOGS_DIR/logs.txt"
done
