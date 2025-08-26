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
        --batch_size 4 \
        --output_dir "$LOGS_DIR" \
        --num_data_workers 3 \
        --max_workers 3 \
        --experiment_id "plots" \
        --gpu "0" \
        --independent_keys \
            PatchProcessing_statistic=RIGID.DINO.05_level=0_patch_size=512_seed=72532 \
            PatchProcessing_statistic=RIGID.CLIP.05_level=0_patch_size=512_seed=72532 \
            PatchProcessing_statistic=RIGID.CLIP.01_level=0_patch_size=512_seed=72532 \
        --patch_divisors 0 \
        --chi2_bins 30 \
        --cdf_bins 400 \
        --dataset_type COCO_STABLE_DIFFUSION_XL_TEST_ONLY \
        --seed 72532 \
        --sample_size 512 \
        --threshold 0.9 \
        --save_histograms 1 \
        --save_independence_heatmaps 1 \
        --pkls_dir pkls_experiments_II \
        --num_samples_per_class -1 \
        --run_id plot_sdxl_jpeg_inference_plot_run_$TIMESTAMP \
        --inference_aug jpeg

    echo "Run complete."
} &> "$LOGS_DIR/logs.txt"
