import argparse
from stat_test import main_multiple_patch_test, main_multiple_wavelet_test
from data_utils import ImageDataset, create_inference_dataset
from torchvision import transforms
from utils import set_seed, plot_sensitivity_specificity_by_patch_size, plot_sensitivity_specificity_by_num_waves

parser = argparse.ArgumentParser(description='Wavelet and Patch Testing Pipeline')
parser.add_argument('--test_type', choices=['multiple_patches', 'multiple_wavelets'], default='multiple_wavelets', help='Choose which type of multiple tests to perform')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for data loading')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing')
parser.add_argument('--cdf_file', type=str, default='patch_population_cdfs_10kb.pkl', help='File name to save/load population CDFs')
parser.add_argument('--reload_cdfs', type=int, choices=[0, 1], default=0, help='Flag to reload precomputed CDFs from file (1 for True, 0 for False)')
parser.add_argument('--save_kdes', type=int, choices=[0, 1], default=0, help='Flag to save KDE plots for real and fake p-values (1 for True, 0 for False)')
parser.add_argument('--ensemble_test', choices=['stouffer', 'rbm'], default='stouffer', help='Type of ensemble test to perform (e.g., stouffer, rbm)')
parser.add_argument('--save_independence_heatmaps', type=int, choices=[0, 1], default=0, help='Flag to save independence test heatmaps (1 for True, 0 for False)')
parser.add_argument('--data_dir_real', type=str, default='data/CelebaHQMaskDataset/train/images_faces', help='Path to the real population dataset')
parser.add_argument('--data_dir_fake_real', type=str, default='data/CelebaHQMaskDataset/test/images_faces', help='Path to the real-fake dataset')
parser.add_argument('--data_dir_fake', type=str, default='data/stable-diffusion-face-dataset/1024/both_faces', help='Path to the fake dataset')
parser.add_argument('--output_dir', type=str, default='logs', help='Path where to save artifacts')
parser.add_argument('--num_samples_per_class', type=int, default=2957, help='Number of samples per class for inference dataset')
parser.add_argument('--num_data_workers', type=int, default=4, help='Number of workers for data loading')
parser.add_argument('--max_workers', type=int, default=32, help='Maximum number of threads for parallel processing')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()


def main():
    set_seed(args.seed)

    # Load datasets
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    real_population_dataset = ImageDataset(args.data_dir_real, transform=transform, labels=0)
    inference_data = create_inference_dataset(args.data_dir_fake_real, args.data_dir_fake, args.num_samples_per_class, classes='both')

    # Prepare inference dataset
    image_paths = [x[0] for x in inference_data]
    labels = [x[1] for x in inference_data]
    inference_dataset = ImageDataset(image_paths, labels, transform=transform)

    if args.test_type == 'multiple_patches':
        thresholds = [0.05, 0.25, 0.5]
        wavelets = ['bior1.1', 'coif1', 'haar', 'sym2']
        patch_sizes = [256, 128, 64, 32, 16]
        results = {}

        for threshold in thresholds:
            results[threshold] = {}
            for wavelet in wavelets:
                results[threshold][wavelet] = {}
                for patch_size in patch_sizes:
                    results[threshold][wavelet][patch_size] = main_multiple_patch_test(
                        real_population_dataset=real_population_dataset,
                        inference_dataset=inference_dataset,
                        test_labels=labels,
                        batch_size=args.batch_size,
                        threshold=threshold,
                        patch_size=patch_size,
                        wavelet=wavelet,
                        cdf_file=args.cdf_file,
                        reload_cdfs=bool(args.reload_cdfs),
                        save_independence_heatmaps=bool(args.save_independence_heatmaps),
                        save_kdes=bool(args.save_kdes),
                        ensemble_test=args.ensemble_test,
                        max_workers=args.max_workers,
                        num_data_workers=args.num_data_workers,
                        output_dir=args.output_dir
                    )
                plot_sensitivity_specificity_by_patch_size(results[threshold][wavelet], wavelet, threshold, args.output_dir)

    elif args.test_type == 'multiple_wavelets':
        thresholds = [0.05, 0.25, 0.5]
        wavelist = ['bior1.1', 'bior3.1', 'bior6.8', 'coif1', 'coif10', 'db1', 'db38', 'haar', 'rbio6.8', 'sym2']

        results = {}

        for threshold in thresholds:
            results[threshold] = {}
            for i in range(1, len(wavelist) + 1):
                results[threshold][i] = main_multiple_wavelet_test(
                    real_population_dataset,
                    inference_dataset,
                    test_labels=labels,
                    batch_size=args.batch_size,
                    threshold=threshold,
                    cdf_file=args.cdf_file,
                    reload_cdfs=bool(args.reload_cdfs),
                    save_independence_heatmaps=bool(args.save_independence_heatmaps),
                    save_kdes=bool(args.save_kdes),
                    ensemble_test=args.ensemble_test,
                    wavelet_list=wavelist[:i],
                    max_workers=args.max_workers,
                    num_data_workers=args.num_data_workers,
                    output_dir=args.output_dir
                )
            plot_sensitivity_specificity_by_num_waves(results[threshold], threshold, args.output_dir)


if __name__ == "__main__":
    main()
