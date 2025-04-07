import os
import argparse
import sys

from tqdm import tqdm
from stat_test import TestType, main_multiple_patch_test
from datasets_factory import DatasetFactory, DatasetType
from data_utils import ImageDataset, create_inference_dataset
from torchvision import transforms
from utils import build_backbones_statistics_list, plot_roc_curve, set_seed

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Argument parser
parser = argparse.ArgumentParser(description='Wavelet and Patch Testing Pipeline')
parser.add_argument('--test_type', choices=['multiple_patches', 'multiple_wavelets'], default='multiple_wavelets', help='Choose which type of multiple tests to perform')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loading')
parser.add_argument('--sample_size', type=int, default=256, help='Sample input size after downscale')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing')
parser.add_argument('--save_histograms', type=int, choices=[0, 1], default=1, help='Flag to save KDE plots for real and fake p-values (1 for True, 0 for False)')
parser.add_argument('--ensemble_test', choices=['manual-stouffer', 'stouffer', 'rbm'], default='manual-stouffer', help='Type of ensemble test to perform')
parser.add_argument('--save_independence_heatmaps', type=int, choices=[0, 1], default=1, help='Flag to save independence test heatmaps (1 for True, 0 for False)')
parser.add_argument('--dataset_type', type=str, default='COCO_STABLE_DIFFUSION_2_768', choices=[e.name for e in DatasetType], help='Type of dataset to use (CelebA, ProGan, COCO_LEAKAGE, COCO, COCO_ALL, PROGAN_FACES_BUT_CELEBA_AS_TRAIN)')
parser.add_argument('--output_dir', type=str, default='logs', help='Path where to save artifacts')
parser.add_argument('--pkls_dir', type=str, default='/data/users/haimzis/rigid_pkls', help='Path where to save pkls')
parser.add_argument('--num_samples_per_class', type=int, default=-1, help='Number of samples per class for inference dataset')
parser.add_argument('--num_data_workers', type=int, default=4, help='Number of workers for data loading')
parser.add_argument('--max_wave_level', type=int, default=4, help='Maximum number of levels in DWT')
parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of threads for parallel processing')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()


# TO REMOVE
for filename in os.listdir(args.output_dir):
    file_path = os.path.join(args.output_dir, filename)
    
    # Check if it is a file (not a subdirectory)
    if os.path.isfile(file_path):
        os.remove(file_path)  # Remove the file\
        print(f"Deleted file: {filename}")


# Define datasets to evaluate: TODO: REMOVE
DATASETS_TO_EVALUATE = [
    'PROGAN_FACES', 'COCO', 'COCO_BIGGAN_256', 'COCO_STABLE_DIFFUSION_XL', 
    'COCO_DALLE3_COCOVAL', 'COCO_SYNTH_MIDJOURNEY_V5', 'COCO_STABLE_DIFFUSION_2',
    'BIGGAN_TEST_ONLY', 'CYCLEGAN_TEST_ONLY', 'GAUGAN_TEST_ONLY', 'PROGAN_TEST_ONLY',
    'SEEINGDARK_TEST_ONLY', 'STYLEGAN_TEST_ONLY', 'CRN_TEST_ONLY', 'DEEPFAKE_TEST_ONLY',
    'IMLE_TEST_ONLY', 'SAN_TEST_ONLY', 'STARGAN_TEST_ONLY', 'STYLEGAN2_TEST_ONLY', 'COCO_STABLE_DIFFUSION_2_768'
]

def main():
    set_seed(args.seed)

    # Get paths dynamically based on dataset_type
    dataset_type_enum = DatasetType[args.dataset_type.upper()]
    paths = dataset_type_enum.get_paths()

    # Dynamically create a subdirectory in pkls_dir based on dataset type
    dataset_pkls_dir = os.path.join(args.pkls_dir, args.dataset_type)
    os.makedirs(dataset_pkls_dir, exist_ok=True)

    # Load datasets
    transform = transforms.Compose([
        transforms.Resize((args.sample_size, args.sample_size)),
        transforms.ToTensor()
    ])

    datasets = DatasetFactory.create_dataset(dataset_type=args.dataset_type, transform=transform)
    real_population_dataset, fake_population_dataset = datasets['train_real'], datasets['train_fake']
    inference_data = create_inference_dataset(paths['test_real']['path'], paths['test_fake']['path'], args.num_samples_per_class, classes='both')

    # Prepare inference dataset
    image_paths = [x[0] for x in inference_data]
    labels = [x[1] for x in inference_data]
    inference_dataset = ImageDataset(image_paths, labels, transform=transform)

    threshold = args.threshold
    # waves = ['bior6.8', 'rbio6.8', 'bior1.1', 'bior3.1', 'sym2', 'haar', 'coif1', 'fourier', 'dct', 'blurness', 'gabor', 'hsv', 'jpeg', 'sift', 'ssim', 'psnr']
    # waves = ['edge5x5', 'smoothing', 'noise', 'sharpness', 'emboss', 'highpass', 'sobel', 'gauss_diff']

    models = ['CONVNEXT', 'DINO', 'BEIT', 'CLIP', 'DEIT', 'RESNET']
    noise_levels = ['01', '05', '10', '50', '75', '100']

    waves = build_backbones_statistics_list(models, noise_levels)
    
    patch_sizes = [256]
    wavelet_levels = [0]
    
    test_id = f"num_waves_{len(waves)}-{min(patch_sizes)}_{max(patch_sizes)}-max_level_{wavelet_levels[-1]}"

    results = main_multiple_patch_test(
            real_population_dataset=real_population_dataset,
            fake_population_dataset=fake_population_dataset,
            inference_dataset=inference_dataset,
            test_labels=labels,
            batch_size=args.batch_size,
            threshold=threshold,
            patch_sizes=patch_sizes,
            waves=waves,
            wavelet_levels=wavelet_levels,
            save_independence_heatmaps=bool(args.save_independence_heatmaps),
            save_histograms=bool(args.save_histograms),
            ensemble_test=args.ensemble_test,
            max_workers=args.max_workers,
            num_data_workers=args.num_data_workers,
            output_dir=args.output_dir,
            pkl_dir=dataset_pkls_dir,
            return_logits=True,
            portion=0.1,
            chi2_bins=10,
            cdf_bins=2000,
            n_trials=75,
            uniform_p_threshold=0.05,
            calibration_auc_threshold=0.4,
            ks_pvalue_abs_threshold=0.3,
            minimal_p_threshold=0.01,
            test_type=TestType.BOTH
        )

    results['labels'] = labels
    plot_roc_curve(results, test_id, args.output_dir)


if __name__ == "__main__":
    sys.setrecursionlimit(2000)
    main()
    ## TODO: REMOVE and leave only main()
    # for dataset in tqdm(DATASETS_TO_EVALUATE, desc="Running evaluations"):
    #     try:
    #         args.dataset_type = dataset
    #         main()
    #     except Exception as e:
    #         print(f"Error with dataset {dataset}: {e}")