import os
import argparse
from data_utils import DatasetFactory, DatasetType, ImageDataset, create_inference_dataset
from torchvision import transforms
from stat_test import DataType, get_unique_id, preprocess_wave
from utils import load_population_histograms, plot_pvalue_histograms, set_seed, save_population_histograms
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Custom Histogram and Testing Pipeline')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for data loading')
parser.add_argument('--sample_size', type=int, default=256, help='Sample input size after downscale')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing')
parser.add_argument('--dataset_type', type=str, default='COCO_ALL', choices=[e.name for e in DatasetType], help='Type of dataset to use')
parser.add_argument('--num_samples_per_class', type=int, default=-1, help='Number of samples per class for inference dataset')
parser.add_argument('--num_data_workers', type=int, default=0, help='Number of workers for data loading')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--pkls_dir', type=str, default='/data/users/haimzis/pkls', help='Path where to save pkls')

args = parser.parse_args()

histograms_stats_dir = os.path.join('histograms_stats', 'self_patch')

def preprocess_and_plot():
    set_seed(args.seed)
    patch_size = 256
    level = 0

    for dataset in ['COCO_LEAKAGE']:#, 'COCO_ALL', 'COCO', 'CelebA', 'ProGan']:
        os.makedirs(os.path.join(histograms_stats_dir, dataset), exist_ok=True)  

        for wavelet in ['bior6.8', 'rbio6.8', 'bior1.1', 'bior3.1', 'sym2', 'haar', 'coif1', 'fourier', 'dct'] + ['blurness', 'gabor', 'hsv', 'jpeg', 'laplacian', 'sift', 'ssim']:
            
            artifact_path = os.path.join(histograms_stats_dir, dataset, f"{dataset}_{wavelet}_statistic.png")
            if os.path.exists(artifact_path):
                continue

            dataset_pkls_dir = os.path.join(args.pkls_dir, dataset)
            
            # Get dataset paths based on dataset_type
            dataset_type_enum = DatasetType[dataset.upper()]
            paths = dataset_type_enum.value

            # Data transforms
            transform = transforms.Compose([transforms.Resize((args.sample_size, args.sample_size)), transforms.ToTensor()])

            # Load datasets
            real_population_dataset, _ = DatasetFactory.create_dataset(dataset_type=dataset, root_dir=paths['train_real'], calib_root_dir=paths['train_fake'], transform=transform)
            inference_data = create_inference_dataset(paths['test_real'], paths['test_fake'], args.num_samples_per_class, classes='fake')

            # Prepare inference dataset
            image_paths = [x[0] for x in inference_data]
            labels = [x[1] for x in inference_data]
            inference_dataset = ImageDataset(image_paths, labels, transform=transform)

            print("Using custom histogram object...")

            # Preprocess the real population dataset
            real_histograms = preprocess_wave(
                real_population_dataset, args.batch_size, wavelet, level, 0, patch_size, dataset_pkls_dir, False, DataType.TRAIN)

            # Preprocess the inference dataset
            inference_histograms = preprocess_wave(inference_dataset, args.batch_size, wavelet, level, 0, patch_size, dataset_pkls_dir, False, DataType.TEST)

            if real_histograms is None or inference_histograms is None: 
                continue
            
            # Plot the histograms of both classes
            plot_pvalue_histograms(
                real_histograms,
                inference_histograms,
                artifact_path,
                title=f"Combined Histogram for Real and Fake Samples - {wavelet}",
                xlabel='statistic values'
            )


if __name__ == "__main__":
    preprocess_and_plot()
