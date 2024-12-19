#!/bin/bash

# Define dataset paths
DATA_DIR_REAL="data/CelebaHQMaskDataset/train/images_faces"
DATA_DIR_FAKE_REAL="data/CelebaHQMaskDataset/test/images_faces"
DATA_DIR_FAKE="data/stable-diffusion-face-dataset/1024/both_faces"

# Base directory for logs
LOGS_BASE_DIR="logs"

# Hard-coded configurations
configs=(
    "--finetune_portion 0.05 --waves haar coif1 sym2 --criteria KS --patch_divisors 0 1 2"
    "--finetune_portion 0.1 --waves fourier dct --criteria N --patch_divisors 1 2 3"
    "--finetune_portion 0.15 --waves haar coif1 sym2 fourier --criteria KS --patch_divisors 0 2 4"
    "--finetune_portion 0.2 --waves haar coif1 sym2 fourier dct --criteria N --patch_divisors 2 3 4"
    "--finetune_portion 0.05 --waves haar coif1 sym2 fourier --criteria KS --patch_divisors 1 3"
    "--finetune_portion 0.1 --waves fourier dct --criteria KS --patch_divisors 0 2"
    "--finetune_portion 0.15 --waves haar coif1 sym2 --criteria KS --patch_divisors 1 2"
    "--finetune_portion 0.2 --waves fourier dct --criteria N --patch_divisors 0 3"
    "--finetune_portion 0.05 --waves haar coif1 sym2 fourier dct --criteria KS --patch_divisors 0 1 2"
    "--finetune_portion 0.1 --waves haar coif1 sym2 --criteria N --patch_divisors 2 3"
    "--finetune_portion 0.15 --waves fourier dct --criteria KS --patch_divisors 1 2 4"
    "--finetune_portion 0.2 --waves haar coif1 sym2 --criteria N --patch_divisors 0 2"
    "--finetune_portion 0.05 --waves fourier dct --criteria KS --patch_divisors 1 2"
    "--finetune_portion 0.1 --waves haar coif1 sym2 fourier dct --criteria N --patch_divisors 2 3 4"
    "--finetune_portion 0.15 --waves haar coif1 sym2 --criteria KS --patch_divisors 0 1 3"
    "--finetune_portion 0.2 --waves fourier dct --criteria KS --patch_divisors 0 3 4"
    "--finetune_portion 0.05 --waves haar coif1 sym2 --criteria N --patch_divisors 1 2 3"
    "--finetune_portion 0.1 --waves fourier dct --criteria KS --patch_divisors 0 1 4"
    "--finetune_portion 0.15 --waves haar coif1 sym2 fourier --criteria N --patch_divisors 2 3"
    "--finetune_portion 0.2 --waves haar coif1 sym2 --criteria KS --patch_divisors 0 1 4"
    "--finetune_portion 0.25 --waves haar coif1 sym2 fourier dct --criteria KS --patch_divisors 1 2 4"
    "--finetune_portion 0.3 --waves haar coif1 sym2 --criteria N --patch_divisors 0 1 3"
    "--finetune_portion 0.05 --waves fourier dct --criteria KS --patch_divisors 2 3"
    "--finetune_portion 0.1 --waves haar coif1 sym2 fourier --criteria N --patch_divisors 0 3"
    "--finetune_portion 0.15 --waves haar coif1 sym2 fourier dct --criteria KS --patch_divisors 1 2 3"
    "--finetune_portion 0.2 --waves fourier dct --criteria N --patch_divisors 0 2 4"
    "--finetune_portion 0.25 --waves haar coif1 sym2 --criteria KS --patch_divisors 1 3"
    "--finetune_portion 0.3 --waves haar coif1 sym2 fourier dct --criteria N --patch_divisors 2 4"
    "--finetune_portion 0.1 --waves haar coif1 sym2 --criteria KS --patch_divisors 0 1 2"
    "--finetune_portion 0.05 --waves fourier dct --criteria N --patch_divisors 1 3 4"
    "--finetune_portion 0.15 --waves haar coif1 sym2 fourier --criteria KS --patch_divisors 0 2"
    "--finetune_portion 0.2 --waves haar coif1 sym2 fourier dct --criteria N --patch_divisors 3 4"
    "--finetune_portion 0.25 --waves fourier dct --criteria KS --patch_divisors 0 1"
    "--finetune_portion 0.3 --waves haar coif1 sym2 --criteria N --patch_divisors 2 3 4"
    "--finetune_portion 0.05 --waves haar coif1 sym2 fourier dct --criteria KS --patch_divisors 0 1 3"
    "--finetune_portion 0.1 --waves haar coif1 sym2 --criteria KS --patch_divisors 2 4"
    "--finetune_portion 0.15 --waves fourier dct --criteria N --patch_divisors 1 2 3"
    "--finetune_portion 0.2 --waves haar coif1 sym2 fourier --criteria KS --patch_divisors 0 3 4"
    "--finetune_portion 0.25 --waves fourier dct --criteria KS --patch_divisors 0 2"
    "--finetune_portion 0.3 --waves haar coif1 sym2 fourier dct --criteria N --patch_divisors 1 3"
    "--finetune_portion 0.1 --waves haar coif1 sym2 --criteria KS --patch_divisors 2 3"
    "--finetune_portion 0.05 --waves haar coif1 sym2 fourier dct --criteria KS --patch_divisors 0 1 2"
    "--finetune_portion 0.15 --waves haar coif1 sym2 fourier --criteria N --patch_divisors 1 4"
    "--finetune_portion 0.2 --waves fourier dct --criteria KS --patch_divisors 2 3 4"
    "--finetune_portion 0.25 --waves haar coif1 sym2 --criteria KS --patch_divisors 0 2"
    "--finetune_portion 0.3 --waves haar coif1 sym2 fourier --criteria N --patch_divisors 3 4"
)


# Loop through hard-coded configurations
for i in "${!configs[@]}"; do
    # Generate timestamped logs directory
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOGS_DIR="$LOGS_BASE_DIR/run_$TIMESTAMP"
    mkdir -p "$LOGS_DIR"

    # Redirect all output to logs.txt
    {
        echo "Starting run $((i + 1))..."
        echo "Command: python executor.py ${configs[$i]}"

        python executor.py \
            --batch_size 256 \
            --sample_size 256 \
            --data_dir_real "$DATA_DIR_REAL" \
            --data_dir_fake_real "$DATA_DIR_FAKE_REAL" \
            --data_dir_fake "$DATA_DIR_FAKE" \
            --output_dir "$LOGS_DIR" \
            --pkls_dir "/data/users/haimzis/pkls" \
            --num_samples_per_class 2957 \
            --num_data_workers 4 \
            --max_wave_level 4 \
            --max_workers 16 \
            --seed 42 \
            ${configs[$i]}

        echo "Run $((i + 1)) complete. Logs saved to $LOGS_DIR/logs.txt"
    } &> "$LOGS_DIR/logs.txt"
done
