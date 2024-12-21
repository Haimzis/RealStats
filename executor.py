import os
import argparse
from urllib.parse import urlparse
from stat_test import main_multiple_patch_test
from data_utils import ImageDataset, create_inference_dataset
from torchvision import transforms
from utils import plot_roc_curve, set_seed
import sys
import mlflow

sys.setrecursionlimit(2000)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Define argument parser
parser = argparse.ArgumentParser(description='Wavelet and Patch Testing Pipeline')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for data loading.')
parser.add_argument('--sample_size', type=int, default=256, help='Sample input size after downscale.')
parser.add_argument('--patch_divisors', type=int, nargs='+', default=[2, 4, 8], help='Divisors to calculate patch sizes as sample_size // 2^i.')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing.')
parser.add_argument('--save_histograms', type=int, choices=[0, 1], default=1, help='Save KDE plots for real and fake p-values.')
parser.add_argument('--ensemble_test', choices=['manual-stouffer', 'stouffer', 'rbm'], default='manual-stouffer', help='Type of ensemble test.')
parser.add_argument('--save_independence_heatmaps', type=int, choices=[0, 1], default=1, help='Save independence test heatmaps.')
parser.add_argument('--data_dir_real', type=str, required=True, help='Path to the real population dataset.')
parser.add_argument('--data_dir_fake_real', type=str, required=True, help='Path to the real-fake dataset.')
parser.add_argument('--data_dir_fake', type=str, required=True, help='Path to the fake dataset.')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save logs and artifacts.')
parser.add_argument('--pkls_dir', type=str, default='/data/users/haimzis/pkls', help='Path where to save pkls.')
parser.add_argument('--num_samples_per_class', type=int, default=2957, help='Number of samples per class for inference dataset.')
parser.add_argument('--num_data_workers', type=int, default=4, help='Number of workers for data loading.')
parser.add_argument('--max_wave_level', type=int, default=4, help='Maximum number of levels in DWT.')
parser.add_argument('--max_workers', type=int, default=16, help='Maximum number of threads for parallel processing.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
parser.add_argument('--waves', type=str, nargs='+', default=['haar', 'coif1', 'sym2', 'fourier', 'dct'], help='List of wavelet types.')
parser.add_argument('--wavelet_levels', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='List of wavelet levels.')
parser.add_argument('--finetune_portion', type=float, default=0.05, help='Portion of the dataset used for finetuning.')
parser.add_argument('--criteria', type=str, choices=['KS', 'N'], required=True, help='Criteria for optimization (KS or N).')
parser.add_argument('--chi2_bins', type=int, default=201, help='Number of bins for chi-square calculations.')
parser.add_argument('--n_trials', type=int, default=50, help='Number of trials for optimization.')
parser.add_argument('--run_id', type=str, required=True, help='Unique identifier for this MLflow run.')

args = parser.parse_args()

def main():
    # Set random seed
    set_seed(args.seed)

    mlflow.set_experiment("independent_stouffer")
    with mlflow.start_run(run_name=args.run_id):
        args.output_dir = urlparse(mlflow.get_artifact_uri()).path
        
        # Load datasets
        transform = transforms.Compose([transforms.Resize((args.sample_size, args.sample_size)), transforms.ToTensor()])
        real_population_dataset = ImageDataset(args.data_dir_real, transform=transform, labels=0)
        inference_data = create_inference_dataset(args.data_dir_fake_real, args.data_dir_fake, args.num_samples_per_class, classes='both')

        # Prepare inference dataset
        image_paths = [x[0] for x in inference_data]
        labels = [x[1] for x in inference_data]
        inference_dataset = ImageDataset(image_paths, labels, transform=transform)

        # Compute patch sizes based on sample_size and patch_divisors
        patch_sizes = [args.sample_size // 2**i for i in args.patch_divisors]

        mlflow.log_params(vars(args))
        mlflow.log_param('patch_sizes', patch_sizes)

        # Run the main test
        results = main_multiple_patch_test(
            real_population_dataset=real_population_dataset,
            inference_dataset=inference_dataset,
            test_labels=labels,
            batch_size=args.batch_size,
            threshold=args.threshold,
            patch_sizes=patch_sizes,
            waves=args.waves,
            wavelet_levels=args.wavelet_levels,
            save_independence_heatmaps=bool(args.save_independence_heatmaps),
            save_histograms=bool(args.save_histograms),
            ensemble_test=args.ensemble_test,
            max_workers=args.max_workers,
            num_data_workers=args.num_data_workers,
            output_dir=args.output_dir,
            pkl_dir=args.pkls_dir,
            return_logits=True,
            portion=args.finetune_portion,
            criteria=args.criteria,
            chi2_bins=args.chi2_bins,
            n_trials=args.n_trials,
            logger=mlflow
        )

        results['labels'] = labels
        roc_auc = plot_roc_curve(results, args.run_id, args.output_dir)
        mlflow.log_metric('AUC', roc_auc)


if __name__ == "__main__":
    main()
