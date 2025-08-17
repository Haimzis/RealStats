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

    python executor_inference_with_dependence.py \
        --ensemble_test "minp" \
        --batch_size 128 \
        --output_dir "$LOGS_DIR" \
        --num_data_workers 2 \
        --max_workers 4 \
        --experiment_id "runtime_analysis" \
        --gpu "2" \
        --statistics_keys \
            PatchProcessing_wavelet=RIGID.BEIT.01_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.BEIT.01_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.CLIP.01_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.CLIP.01_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.DINO.01_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.DINO.01_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.BEIT.05_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.BEIT.05_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.CLIP.05_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.CLIP.05_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.DINO.05_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.DINO.05_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.BEIT.10_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.BEIT.10_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.CLIP.10_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.RESNET.50_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.RESNET.50_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.CLIP.75_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.CLIP.75_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.DINO.75_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.DINO.75_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.RESNET.75_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.RESNET.75_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.CLIP.100_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.CLIP.100_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.DINO.100_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.DINO.100_level=0_patch_size=512_seed=72532 \
            PatchProcessing_wavelet=RIGID.RESNET.100_level=0_patch_size=256_seed=72532 \
            PatchProcessing_wavelet=RIGID.RESNET.100_level=0_patch_size=512_seed=72532\
        --patch_divisors 0 1 \
        --chi2_bins 20 \
        --cdf_bins 400 \
        --dataset_type COCO_STABLE_DIFFUSION_XL_TEST_ONLY \
        --seed 72532 \
        --sample_size 512 \
        --threshold 0.9 \
        --save_histograms 1 \
        --save_independence_heatmaps 0 \
        --pkls_dir pkls_experiments_II \
        --p_threshold 0.05 \
        --num_samples_per_class -1 \
        --run_id plot_sdxl_inference_independence_$TIMESTAMP

    echo "Run complete."
} &> "$LOGS_DIR/logs.txt"
