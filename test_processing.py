import os
import argparse
import re
from torchvision import transforms
from tqdm import tqdm
from datasets_factory import DatasetFactory, DatasetType
from stat_test import DataType, generate_combinations, patch_parallel_preprocess
from utils import build_backbones_statistics_list, plot_pvalue_histograms, set_seed

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description='Custom Histogram and Testing Pipeline')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loading')
parser.add_argument('--sample_size', type=int, default=256, help='Sample input size after downscale')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing')
parser.add_argument('--dataset_type', type=str, default='COCO_ALL', choices=[e.name for e in DatasetType], help='Type of dataset to use')
parser.add_argument('--num_samples_per_class', type=int, default=-1, help='Number of samples per class for inference dataset')
parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of threads for parallel processing')
parser.add_argument('--num_data_workers', type=int, default=4, help='Number of workers for data loading')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--pkls_dir', type=str, default='rigid_pkls', help='Path where to save pkls')

args = parser.parse_args()

histograms_stats_dir = os.path.join('histograms_stats', '27.3_CLS')


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
    # waves = ['bior6.8', 'rbio6.8', 'bior1.1', 'bior3.1', 'sym2', 'haar', 'coif1', 'fourier', 'dct'] + ['blurness', 'hsv', 'jpeg']
    
    models = ['CONVNEXT', 'DINO', 'BEIT', 'CLIP', 'DEIT', 'RESNET']
    noise_levels = ['01', '05', '10', '50', '75', '100']
    # waves = build_backbones_statistics_list(models, noise_levels)

    waves = ['LatentNoiseCriterion']
    
    
    for dataset in tqdm([dataset.name for dataset in DatasetType], desc="Datasets", unit="dataset"):
        os.makedirs(os.path.join(histograms_stats_dir, dataset), exist_ok=True)  

        dataset_pkls_dir = os.path.join(args.pkls_dir, dataset)
        
        if not os.path.exists(dataset_pkls_dir):
            os.makedirs(dataset_pkls_dir, exist_ok=True)

        # Data transforms
        transform = transforms.Compose([transforms.Resize((args.sample_size, args.sample_size)), transforms.ToTensor()])

        # Load datasets
        datasets = DatasetFactory.create_dataset(dataset_type=dataset, transform=transform)
        real_population_dataset = datasets['train_real']
        inference_dataset = datasets['test_fake']

        stat_combinations = generate_combinations(patch_sizes, waves, levels)

        # Preprocess the real population dataset
        real_histograms = patch_parallel_preprocess(
            real_population_dataset, args.batch_size, stat_combinations, args.max_workers, args.num_data_workers, dataset_pkls_dir, True, DataType.TRAIN
        )
        # Preprocess the inference dataset
        inference_histograms = patch_parallel_preprocess(
            inference_dataset, args.batch_size, stat_combinations, args.max_workers, args.num_data_workers, dataset_pkls_dir, False, DataType.TEST
        )
        
        for key in real_histograms.keys():
            try:
                wave = extract_patch_processing_args(key)['wavelet']
                if real_histograms[key] is None or inference_histograms[key] is None: 
                    continue
        
                artifact_path = os.path.join(histograms_stats_dir, dataset, f"{wave}_statistic.png")

                # Plot the histograms of both classes
                plot_pvalue_histograms(
                    real_histograms[key],
                    inference_histograms[key],
                    artifact_path,
                    title=f"Combined Histogram for Real and Fake Samples - {wave}",
                    xlabel='statistic values'
                )

            except:
                pass


if __name__ == "__main__":
    preprocess_and_plot()

