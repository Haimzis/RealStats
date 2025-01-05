import os
import argparse
from stat_test import main_multiple_patch_test
from data_utils import ImageDataset, create_inference_dataset
from torchvision import transforms
from utils import plot_roc_curve, set_seed

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Wavelet anqd Patch Testing Pipeline')
parser.add_argument('--test_type', choices=['multiple_patches', 'multiple_wavelets'], default='multiple_wavelets', help='Choose which type of multiple tests to perform')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for data loading')
parser.add_argument('--sample_size', type=int, default=256, help='Sample input size after downscale')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing')
parser.add_argument('--histograms_file', type=str, default='patch_population_histograms_10kb.pkl', help='File name to save/load population histograms')
parser.add_argument('--reload_population_histograms', type=int, choices=[0, 1], default=1, help='Flag to reload precomputed population histograms from file (1 for True, 0 for False)')
parser.add_argument('--save_histograms', type=int, choices=[0, 1], default=1, help='Flag to save KDE plots for real and fake p-values (1 for True, 0 for False)')
parser.add_argument('--ensemble_test', choices=['manual-stouffer', 'stouffer', 'rbm'], default='manual-stouffer', help='Type of ensemble test to perform (e.g., stouffer, rbm)')
parser.add_argument('--save_independence_heatmaps', type=int, choices=[0, 1], default=1, help='Flag to save independence test heatmaps (1 for True, 0 for False)')
parser.add_argument('--data_dir_real', type=str, default='data/CelebaHQMaskDataset/train/images_faces', help='Path to the real population dataset')
parser.add_argument('--data_dir_fake_real', type=str, default='data/CelebaHQMaskDataset/test/images_faces', help='Path to the real-fake dataset')
parser.add_argument('--data_dir_fake', type=str, default='data/stable-diffusion-face-dataset/1024/both_faces', help='Path to the fake dataset')
parser.add_argument('--output_dir', type=str, default='logs', help='Path where to save artifacts')
parser.add_argument('--pkls_dir', type=str, default='/data/users/haimzis/pkls', help='Path where to save pkls')
parser.add_argument('--num_samples_per_class', type=int, default=2957, help='Number of samples per class for inference dataset')
parser.add_argument('--num_data_workers', type=int, default=1, help='Number of workers for data loading')
parser.add_argument('--max_wave_level', type=int, default=4, help='Maximum number of levels in DWT')
parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of threads for parallel processing')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()


def main():
    set_seed(args.seed)

    # Load datasets
    transform = transforms.Compose([transforms.Resize((args.sample_size, args.sample_size)), transforms.ToTensor()])
    real_population_dataset = ImageDataset(args.data_dir_real, transform=transform, labels=0)
    inference_data = create_inference_dataset(args.data_dir_fake_real, args.data_dir_fake, args.num_samples_per_class, classes='both')

    # Prepare inference dataset
    image_paths = [x[0] for x in inference_data]
    labels = [x[1] for x in inference_data]
    inference_dataset = ImageDataset(image_paths, labels, transform=transform)

    threshold = 0.05
    waves = ['bior6.8', 'rbio6.8', 'bior1.1', 'bior3.1', 'sym2'] #['haar', 'coif1', 'sym2', 'fourier', 'dct']
    patch_sizes = [256, 128, 64] #[args.sample_size // 2**i for i in range(3)]
    wavelet_levels= [0, 1, 2, 3, 4] # range(3)
    
    test_id = f"num_waves_{len(waves)}-{min(patch_sizes)}_{max(patch_sizes)}-max_level_{wavelet_levels[-1]}"

    results = main_multiple_patch_test(
            real_population_dataset=real_population_dataset,
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
            pkl_dir=args.pkls_dir,
            return_logits=True,
            portion=0.2,
            chi2_bins=10,
            n_trials=100,
            uniform_p_threshold=0.05
        )

    results['labels'] = labels
    # plot_sensitivity_specificity_by_patch_size(results[threshold][wavelet], wavelet, threshold, args.output_dir)
    plot_roc_curve(results, test_id, args.output_dir)


if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(2000)
    main()
