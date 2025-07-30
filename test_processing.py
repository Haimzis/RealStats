import os
import argparse
import re
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from datasets_factory import DatasetFactory, DatasetType
from stat_test import DataType, generate_combinations, patch_parallel_preprocess
from utils import build_backbones_statistics_list, plot_pvalue_histograms, set_seed
from data_utils import ImageDataset, create_inference_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Custom Histogram and Testing Pipeline')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for data loading')
parser.add_argument('--sample_size', type=int, default=512, help='Sample input size after downscale')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing')
parser.add_argument('--dataset_type', type=str, default='COCO_ALL', choices=[e.name for e in DatasetType], help='Type of dataset to use')
parser.add_argument('--num_samples_per_class', type=int, default=-1, help='Number of samples per class for inference dataset')
parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of threads for parallel processing')
parser.add_argument('--num_data_workers', type=int, default=2, help='Number of workers for data loading')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--pkls_dir', type=str, default='pkls_adaptability', help='Path where to save pkls')

args = parser.parse_args()

histograms_stats_dir = os.path.join('histograms_stats', '29.07_adaptability')


def extract_patch_processing_args(key: str):
    match = re.match(r"PatchProcessing_wavelet=([\w.]+)_level=(\d+)_patch_size=(\d+)", key)
    
    if match:
        wavelet, level, patch_size = match.groups()
        return {
            'wavelet': wavelet,
            'level': int(level),
            'patch_size': int(patch_size),
        }


def preprocess_and_plot():
    set_seed(args.seed)
    patch_sizes = [args.sample_size]
    levels = [0]
    # waves = ['bior6.8', 'rbio6.8', 'bior1.1', 'bior3.1', 'sym2', 'haar', 'coif1', 'fourier', 'jpeg', 'hsv']
    models = ['DINO', 'BEIT', 'CLIP', 'RESNET']
    # noise_levels = ['01', '05', '10', '50', '75', '100']
    noise_levels = ['01', '05', '10']
    waves = build_backbones_statistics_list(models, noise_levels)
    # waves += ['LatentNoiseCriterion']

    # dataset_types = [
    #     'BIGGAN_TEST_ONLY', 'CYCLEGAN_TEST_ONLY', 'GAUGAN_TEST_ONLY', 'PROGAN_TEST_ONLY',
    #     'SEEINGDARK_TEST_ONLY', 'STYLEGAN_TEST_ONLY', 'CRN_TEST_ONLY', 'DEEPFAKE_TEST_ONLY',
    #     'IMLE_TEST_ONLY', 'SAN_TEST_ONLY', 'STARGAN_TEST_ONLY', 'STYLEGAN2_TEST_ONLY',
    #     'CELEBA_TEST_ONLY', 'COCO_TEST_ONLY', 'COCO_BIGGAN_256_TEST_ONLY',
    #     'COCO_STABLE_DIFFUSION_XL_TEST_ONLY', 'COCO_DALLE3_COCOVAL_TEST_ONLY',
    #     'COCO_SYNTH_MIDJOURNEY_V5_TEST_ONLY', 'COCO_STABLE_DIFFUSION_2_TEST_ONLY'
    # ]

    dataset_types = ['COCO_STABLE_DIFFUSION_1_4_TEST_ONLY', 'GAUGAN_TEST_ONLY']

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

        stat_combinations = generate_combinations(patch_sizes, waves, levels)

        # Process both real and fake from inference set
        real_histograms = patch_parallel_preprocess(
            real_dataset, args.batch_size, stat_combinations, args.max_workers,
            args.num_data_workers, dataset_pkls_dir, True, DataType.CALIB
        )

        fake_histograms = patch_parallel_preprocess(
            fake_dataset, args.batch_size, stat_combinations, args.max_workers,
            args.num_data_workers, dataset_pkls_dir, True, DataType.TEST
        )

        for key in real_histograms.keys():
            try:
                wave = extract_patch_processing_args(key)['wavelet']
                if real_histograms[key] is None or fake_histograms[key] is None:
                    continue

                artifact_path = os.path.join(histograms_stats_dir, dataset, f"{wave}_statistic.png")

                # plot_pvalue_histograms(
                #     real_histograms[key],
                #     fake_histograms[key],
                #     artifact_path,
                #     title=f"Histogram: Real vs Fake",
                #     xlabel='statistic values'
                # )

                plt.figure(figsize=(8, 8))
                plt.hist(real_histograms[key], bins=50, alpha=0.6, label="Real", color='tab:blue', density=True)
                plt.hist(fake_histograms[key], bins=50, alpha=0.6, label="Fake", color='tab:orange', density=True)
                plt.legend(loc='upper left', fontsize=18, frameon=True)
                plt.tight_layout()
                plt.savefig(artifact_path.replace('.png', '.svg'))
                plt.close()


            except Exception as e:
                print(f"Skipping {key} due to error: {e}")


if __name__ == "__main__":
    preprocess_and_plot()

