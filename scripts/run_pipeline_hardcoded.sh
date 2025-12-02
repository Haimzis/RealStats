#!/bin/bash

# Base directory for logs
LOGS_BASE_DIR="logs"

# Generate timestamped logs directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGS_DIR="$LOGS_BASE_DIR/run_$TIMESTAMP"
mkdir -p "$LOGS_DIR"

# Execute pipeline with hardcoded arguments
python pipeline.py \
    --batch_size 64 \
    --sample_size 512 \
    --threshold 0.05 \
    --output_dir "$LOGS_DIR" \
    --num_samples_per_class -1 \
    --num_data_workers 6 \
    --max_workers 4 \
    --run_id "pipeline_run_$TIMESTAMP" \
    --gpu "0,1,2" \
    --statistics RIGID.DINO.05 RIGID.DINO.10 RIGID.DINOV3.VITS16.05 RIGID.DINOV3.VITS16.10 RIGID.CLIPOPENAI.05 RIGID.CLIPOPENAI.10 RIGID.CONVNEXT.05 RIGID.CONVNEXT.10 \
    --ensemble_test minp \
    --patch_divisors 0 \
    --chi2_bins 15 \
    --dataset_type ALL \
    --pkls_dir pkls/AIStats/new_stats \
    --cdf_bins 400 \
    --ks_pvalue_abs_threshold 0.45 \
    --cremer_v_threshold 0.07 \
    --experiment_id AIStats2/all_splits_minp \
    --preferred_statistics RIGID.DINO.05 RIGID.CLIPOPENAI.05 RIGID.DINO.10 RIGID.CLIPOPENAI.10 \
    --seed 38
