#!/bin/bash

# Base directory for logs
LOGS_BASE_DIR="logs"

# Generate timestamped logs directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGS_DIR="$LOGS_BASE_DIR/run_$TIMESTAMP"
mkdir -p "$LOGS_DIR"

# Redirect all output (stdout + stderr) to logs.txt
{
    echo "Starting run..."

    python executor_inference.py \
        --ensemble_test "minp" \
        --batch_size 16 \
        --output_dir "$LOGS_DIR" \
        --num_data_workers 3 \
        --max_workers 2 \
        --experiment_id "plots" \
        --gpu "0,1,2,3" \
        --independent_keys \
            PatchProcessing_statistic=RIGID.DINO.05_level=0_patch_size=512_seed=57436 \
            PatchProcessing_statistic=RIGID.CLIP.05_level=0_patch_size=512_seed=57436 \
        --patch_divisors 0 \
        --chi2_bins 10 \
        --cdf_bins 400 \
        --dataset_type GAUGAN_TEST_ONLY \
        --seed 57436 \
        --sample_size 512 \
        --threshold 0.9 \
        --save_histograms 1 \
        --save_independence_heatmaps 1 \
        --pkls_dir pkls_experiments_II \
        --num_samples_per_class -1 \
        --run_id plot_gaugan_inference_plot_run_$TIMESTAMP

    echo "Run complete."
} &> "$LOGS_DIR/logs.txt"
