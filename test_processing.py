import os
import argparse
import re
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from datasets_factory import DatasetFactory, DatasetType
from stat_test import DataType, generate_combinations, patch_parallel_preprocess
from utils import build_backbones_statistics_list, plot_pvalue_histograms, set_seed
from data_utils import ImageDataset, create_inference_dataset
from torch.utils.data import ConcatDataset
from statistics_factory import STATISTIC_HISTOGRAMS


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Custom Histogram and Testing Pipeline')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading')
parser.add_argument('--sample_size', type=int, default=512, help='Sample input size after downscale')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing')
parser.add_argument('--dataset_type', type=str, default='COCO_ALL', choices=[e.name for e in DatasetType], help='Type of dataset to use')
parser.add_argument('--num_samples_per_class', type=int, default=-1, help='Number of samples per class for inference dataset')
parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of threads for parallel processing')
parser.add_argument('--num_data_workers', type=int, default=2, help='Number of workers for data loading')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument('--pkls_dir', type=str, default='pkls/AIStats/new_stats', help='Path where to save pkls')

args = parser.parse_args()

histograms_stats_dir = os.path.join('histograms_stats', '13.09_adaptability')


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
    # models = ['DINO', 'BEIT', 'CLIP', 'RESNET']
    # # noise_levels = ['01', '05', '10', '50', '75', '100']
    # noise_levels = ['01', '05', '10']
    # statistics = build_backbones_statistics_list(models, noise_levels)
    # statistics += ['LatentNoiseCriterion']
    statistics = [k for k in STATISTIC_HISTOGRAMS if k.startswith("RIGID.") and any(k.endswith(suffix) for suffix in [".05", ".10"])]

    # dataset_types = [
    #     'BIGGAN_TEST_ONLY', 'CYCLEGAN_TEST_ONLY', 'GAUGAN_TEST_ONLY', 'PROGAN_TEST_ONLY',
    #     'SEEINGDARK_TEST_ONLY', 'STYLEGAN_TEST_ONLY', 'CRN_TEST_ONLY', 'DEEPFAKE_TEST_ONLY',
    #     'IMLE_TEST_ONLY', 'SAN_TEST_ONLY', 'STARGAN_TEST_ONLY', 'STYLEGAN2_TEST_ONLY',
    #     'CELEBA_TEST_ONLY', 'COCO_TEST_ONLY', 'COCO_BIGGAN_256_TEST_ONLY',
    #     'COCO_STABLE_DIFFUSION_XL_TEST_ONLY', 'COCO_DALLE3_COCOVAL_TEST_ONLY',
    #     'COCO_SYNTH_MIDJOURNEY_V5_TEST_ONLY', 'COCO_STABLE_DIFFUSION_2_TEST_ONLY'
    # ]

    # Choose your datasets here
    dataset_types = [
        member.name for member in DatasetType
        if "MANIFOLD_BIAS" in member.name and "GROUP_LEAKAGE" in member.name
    ]


    for dataset in tqdm(dataset_types, desc="Datasets", unit="dataset"):
        os.makedirs(os.path.join(histograms_stats_dir, dataset), exist_ok=True)  

        transform = transforms.Compose([
            transforms.Resize((args.sample_size, args.sample_size)),
            transforms.ToTensor()
        ])

        dataset_type_enum = DatasetType[dataset.upper()]
        paths = dataset_type_enum.get_paths()

        datasets = DatasetFactory.create_dataset(dataset_type=dataset, transform=transform)
        reference_dataset = datasets['reference_real']
        test_real_dataset = datasets['test_real']
        test_fake_dataset = datasets['test_fake']

        # Build inference dataset and labels
        inference_dataset = ConcatDataset([test_real_dataset, test_fake_dataset])
        labels = [0] * len(test_real_dataset) + [1] * len(test_fake_dataset)

        stat_combinations = generate_combinations(patch_sizes, statistics, levels)

        # Run histogram generation on real
        real_histograms = patch_parallel_preprocess(
            test_real_dataset, args.batch_size, stat_combinations, args.max_workers,
            args.num_data_workers, args.pkls_dir, DataType.CALIB, seed=args.seed
        )

        # Run histogram generation on fake
        fake_histograms = patch_parallel_preprocess(
            test_fake_dataset, args.batch_size, stat_combinations, args.max_workers,
            args.num_data_workers, args.pkls_dir, DataType.TEST, seed=args.seed
        )
        
        for key in real_histograms.keys():
            try:
                statistic = extract_patch_processing_args(key)['statistic']
                if real_histograms[key] is None or fake_histograms[key] is None:
                    continue

                artifact_path = os.path.join(histograms_stats_dir, dataset, f"{statistic}_statistic.png")

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

