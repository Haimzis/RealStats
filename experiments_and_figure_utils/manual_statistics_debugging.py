import os
import argparse
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
from datasets_factory import DatasetFactory, DatasetType
from stat_test import DataType, TestType, calculate_pvals_from_cdf, generate_combinations, patch_parallel_preprocess, perform_ensemble_testing
from utils import build_backbones_statistics_list, compute_cdf, plot_pvalue_histograms, set_seed
from data_utils import ImageDataset, create_inference_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0 1 2 3"

parser = argparse.ArgumentParser(description='Custom Histogram and Testing Pipeline')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loading')
parser.add_argument('--sample_size', type=int, default=512, help='Sample input size after downscale')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing')
parser.add_argument('--dataset_type', type=str, default='COCO_ALL', choices=[e.name for e in DatasetType], help='Type of dataset to use')
parser.add_argument('--num_samples_per_class', type=int, default=-1, help='Number of samples per class for inference dataset')
parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of threads for parallel processing')
parser.add_argument('--num_data_workers', type=int, default=2, help='Number of workers for data loading')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--pkls_dir', type=str, default='pkls_adaptability', help='Path where to save pkls')

args = parser.parse_args()

histograms_stats_dir = os.path.join('histograms_stats', '29.07_adaptability')


def extract_patch_processing_args(key: str):
    match = re.match(r"PatchProcessing_statistic=([\w.]+)_level=(\d+)_patch_size=(\d+)", key)

    if match:
        statistic, level, patch_size = match.groups()
        return {
            'statistic': statistic,
            'level': int(level),
            'patch_size': int(patch_size),
        }


def preprocess_and_plot():
    set_seed(args.seed)
    patch_sizes = [args.sample_size]
    levels = [0]
    # statistics = ['bior6.8', 'rbio6.8', 'bior1.1', 'bior3.1', 'sym2', 'haar', 'coif1', 'fourier', 'jpeg', 'hsv']
    models = ['DINO', 'BEIT', 'CLIP', 'RESNET']
    # noise_levels = ['01', '05', '10', '50', '75', '100']
    noise_levels = ['01', '05', '10']
    statistics = build_backbones_statistics_list(models, noise_levels)
    # statistics += ['LatentNoiseCriterion']

    # dataset_types = [
    #     'BIGGAN_TEST_ONLY', 'CYCLEGAN_TEST_ONLY', 'GAUGAN_TEST_ONLY', 'PROGAN_TEST_ONLY',
    #     'SEEINGDARK_TEST_ONLY', 'STYLEGAN_TEST_ONLY', 'CRN_TEST_ONLY', 'DEEPFAKE_TEST_ONLY',
    #     'IMLE_TEST_ONLY', 'SAN_TEST_ONLY', 'STARGAN_TEST_ONLY', 'STYLEGAN2_TEST_ONLY',
    #     'CELEBA_TEST_ONLY', 'COCO_TEST_ONLY', 'COCO_BIGGAN_256_TEST_ONLY',
    #     'COCO_STABLE_DIFFUSION_XL_TEST_ONLY', 'COCO_DALLE3_COCOVAL_TEST_ONLY',
    #     'COCO_SYNTH_MIDJOURNEY_V5_TEST_ONLY', 'COCO_STABLE_DIFFUSION_2_TEST_ONLY'
    # ]

    dataset_types = [
        'COCO_STABLE_DIFFUSION_1_4_TEST_ONLY',
        'GAUGAN_TEST_ONLY'
        ]

    for dataset in tqdm(dataset_types, desc="Datasets", unit="dataset"):
        os.makedirs(os.path.join(histograms_stats_dir, dataset), exist_ok=True)  
        dataset_pkls_dir = os.path.join(args.pkls_dir, dataset)
        os.makedirs(dataset_pkls_dir, exist_ok=True)

        transform = transforms.Compose([
            transforms.Resize((args.sample_size, args.sample_size)),
            transforms.ToTensor()
        ])

        dataset_type_enum = DatasetType[dataset.upper()]
        paths = dataset_type_enum.get_paths()

        # Create inference dataset
        inference_data = create_inference_dataset(
            paths['test_real']['path'],
            paths['test_fake']['path'],
            args.num_samples_per_class,
            classes='both'
        )

        # Split into real and fake
        real_paths = [x[0] for x in inference_data if x[1] == 0]
        fake_paths = [x[0] for x in inference_data if x[1] == 1]

        real_dataset = ImageDataset(real_paths, [0] * len(real_paths), transform=transform)
        fake_dataset = ImageDataset(fake_paths, [1] * len(fake_paths), transform=transform)

        stat_combinations = generate_combinations(patch_sizes, statistics, levels)

        # Process both real and fake from inference set
        real_histograms = patch_parallel_preprocess(
            real_dataset, args.batch_size, stat_combinations, args.max_workers,
            args.num_data_workers, dataset_pkls_dir, DataType.CALIB
        )

        fake_histograms = patch_parallel_preprocess(
            fake_dataset, args.batch_size, stat_combinations, args.max_workers,
            args.num_data_workers, dataset_pkls_dir, DataType.TEST
        )

        # Manifold Bias
        csv_filename = os.path.join(args.pkls_dir, f'{dataset}.csv')
        external_df = pd.read_csv(csv_filename)

        external_real = external_df[external_df['label'] == 0].reset_index(drop=True)
        external_fake = external_df[external_df['label'] == 1].reset_index(drop=True)

        external_real_sorted = external_real.set_index('image_path').loc[real_paths].reset_index()
        external_fake_sorted = external_fake.set_index('image_path').loc[fake_paths].reset_index()

        # 2. Extract criterion values as lists or NumPy arrays
        real_criteria = external_real_sorted['criterion'].tolist()
        fake_criteria = external_fake_sorted['criterion'].tolist()

        # 3. Insert into histograms under the 'ManifoldBias' key
        real_histograms['PatchProcessing_statistic=ManifoldBias_level=0_patch_size=512_seed=42'] = np.array([np.array([v]) for v in real_criteria])
        fake_histograms['PatchProcessing_statistic=ManifoldBias_level=0_patch_size=512_seed=42'] = np.array([np.array([v]) for v in fake_criteria])
        
        # for key in real_histograms.keys():
        #     try:
        #         statistic = extract_patch_processing_args(key)['statistic']
        #         if real_histograms[key] is None or fake_histograms[key] is None:
        #             continue

        #         artifact_path = os.path.join(histograms_stats_dir, dataset, f"{statistic}_statistic.png")

        #         # plot_pvalue_histograms(
        #         #     real_histograms[key],
        #         #     fake_histograms[key],
        #         #     artifact_path,
        #         #     title=f"Histogram: Real vs Fake",
        #         #     xlabel='statistic values'
        #         # )

        #         plt.figure(figsize=(8, 8))
        #         plt.hist(real_histograms[key], bins=50, alpha=0.6, label="Real", color='tab:blue', density=True)
        #         plt.hist(fake_histograms[key], bins=50, alpha=0.6, label="Fake", color='tab:orange', density=True)
        #         plt.legend(loc='upper left', fontsize=18, frameon=True)
        #         plt.tight_layout()
        #         plt.savefig(artifact_path.replace('.png', '.svg'))
        #         plt.close()


        #     except Exception as e:
        #         print(f"Skipping {key} due to error: {e}")


        # ----------------------
        # Choose stat keys manually or by rule
        selected_keys = [
            key for key in real_histograms.keys()
            if 'ManifoldBias' in key or 'RIGID' in key
        ]

        # ----------------------
        # Prepare real and fake values for selected keys
        real_hist_selected = {k: [h.squeeze() for h in real_histograms[k]] for k in selected_keys}
        fake_hist_selected = {k: [h.squeeze() for h in fake_histograms[k]] for k in selected_keys}

        # ----------------------
        # Compute reference CDFs from real histograms only
        real_cdfs = {
            k: compute_cdf(real_hist_selected[k], bins=500, test_id=k)
            for k in selected_keys
        }

        # ----------------------
        # Concatenate real and fake histograms for inference evaluation
        inference_histogram = {
            k: real_hist_selected[k] + fake_hist_selected[k]
            for k in selected_keys
        }

        # ----------------------
        # Calculate p-values for inference samples (real + fake)
        inference_pvals = calculate_pvals_from_cdf(real_cdfs, inference_histogram, DataType.TEST.name, TestType.BOTH)
        inference_pvals = np.clip(inference_pvals, 0, 1)

        # ----------------------
        # Perform ensemble test across all selected keys
        ensemble_test = 'minp'
        scores, ensembled_pvals = perform_ensemble_testing(inference_pvals, ensemble_test=ensemble_test, plot=False)

        # ----------------------
        # Predictions (you can later evaluate them if needed)
        threshold = 0.05
        predictions = [1 if p < threshold else 0 for p in ensembled_pvals]

        test_labels = [0] * len(real_paths) + [1] * len(fake_paths)
        
        plot_pvalue_histograms(
            [p for p, l in zip(ensembled_pvals, test_labels) if l == 0],
            [p for p, l in zip(ensembled_pvals, test_labels) if l == 1],
            f"histogram_plot_{patch_sizes}_{ensemble_test}_alpha_{threshold}_{dataset}.png",
            "Histogram of P-values"
        )

if __name__ == "__main__":
    preprocess_and_plot()

