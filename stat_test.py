from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
from data_utils import ImageDataset, PatchDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from statsmodels.stats.multitest import multipletests
from processing.histograms import FourierNormHistogram, WaveletNormHistogram
from utils import calculate_p_value_for_sample, compute_cdf, compute_cdfs, create_inference_dataset, evaluate_predictions, load_population_cdfs, plot_kdes, plot_pvalues_vs_bh_threshold, save_independence_test_heatmaps, save_population_cdfs, set_seed
from scipy.stats import norm, combine_pvalues
import os


def preprocess_wavelet(dataset, batch_size, wavelet, wavelet_level):
    """Preprocess the dataset for a single wavelet level and wavelet type using NormHistogram."""
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    if wavelet in ['bior1.1', 'bior3.1', 'bior6.8', 'coif1', 'coif10', 'db1', 'db38', 'haar', 'rbio6.8', 'sym2']:
        histogram_generator = WaveletNormHistogram(selected_indices=[wavelet_level], wave=wavelet)
    elif wavelet == 'fourier':
        histogram_generator = FourierNormHistogram()
    else: 
        print('Invalid wave.')
        exit(-1)
    histograms = histogram_generator.create_histogram(data_loader)
    return histograms


def wave_parallel_preprocess(dataset, batch_size, wavelet_list, wavelet_levels):
    """Preprocess the dataset for multiple wavelet levels and wavelet types in parallel."""
    results = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_level_wave = {
            executor.submit(preprocess_wavelet, dataset, batch_size, wave, level): (wave, level)
            for wave in wavelet_list
            for level in wavelet_levels
        }
        
        for future in as_completed(future_to_level_wave):
            wave, level = future_to_level_wave[future]
            try:
                results[(wave, level)] = future.result()
            except Exception as exc:
                print(f"Wavelet {wave}, Level {level}, generated an exception: {exc}")
    
    return results


def patch_parallel_preprocess(original_dataset, batch_size, wavelet, wavelet_level, patch_size, num_patches):
    """Preprocess the dataset for a specific wavelet, level, and patch in parallel."""
    results = {}

    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_level_wave = {
            executor.submit(preprocess_wavelet, PatchDataset(original_dataset, patch_size, patch_idx),
                             batch_size, wavelet, wavelet_level): (patch_idx)
            for patch_idx in range(num_patches)
        }
        
        for future in as_completed(future_to_level_wave):
            patch_idx = future_to_level_wave[future]
            try:
                results[patch_idx] = future.result()
            except Exception as exc:
                print(f"Patch {patch_idx}, generated an exception: {exc}")
    
    return results


def fdr_classification(pvalues, threshold=0.05):
    """Perform KS test and apply FDR correction using precomputed CDFs."""
    _, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh', alpha=threshold)
    return int(np.any(pvals_corrected < threshold)) 


def calculate_cdfs_pvalues(real_population_cdfs, input_samples_values):
    p_values_per_test = []

    for wavelet_descriptor in sorted(real_population_cdfs.keys()):
        # Retrieve precomputed CDF for the population
        population_cdf_info = real_population_cdfs[wavelet_descriptor]

        # Calculate p-value for the input sample using precomputed CDF
        sample = input_samples_values[wavelet_descriptor]
        p_value = calculate_p_value_for_sample(sample, population_cdf_info)
        p_values_per_test.append(p_value)
    return p_values_per_test # Return True if any corrected p-value is below the threshold


def main(real_population_dataset, inference_dataset, test_labels=None, batch_size=128, threshold=0.05, patch_size=(32, 32), save_independence_heatmaps=False, wavelet='haar', cdf_file="patch_population_cdfs_10kb.pkl", reload_cdfs=False, save_kdes=False):
    wavelet_level = 0

    # Determine the total number of patches per image
    example_image, _ = real_population_dataset[0]
    h_patches = example_image.shape[1] // patch_size[0]
    w_patches = example_image.shape[2] // patch_size[1]
    num_patches = h_patches * w_patches

    if reload_cdfs and os.path.exists(cdf_file):
        print(f"Loading precomputed population CDFs from {cdf_file}")
        real_population_cdfs = load_population_cdfs(cdf_file)
    else:
        # Generate a single cumulative histogram for each patch index across all images in the real population dataset
        real_population_histograms_per_patch = patch_parallel_preprocess(
            real_population_dataset, batch_size, wavelet, wavelet_level, patch_size, num_patches
        )
        # Compute CDFs for each patch's cumulative histogram
        real_population_cdfs = {patch_idx: compute_cdf(histogram) for patch_idx, histogram in real_population_histograms_per_patch.items()}
        
        print(f"Saving population CDFs to {cdf_file}")
        save_population_cdfs(real_population_cdfs, cdf_file)

    # Preprocess input samples (inference dataset) across all patches
    inference_histograms_per_patch = patch_parallel_preprocess(
        inference_dataset, batch_size, wavelet, wavelet_level, patch_size, num_patches
    )

    input_samples = [dict(zip(inference_histograms_per_patch.keys(), values)) for values in zip(*inference_histograms_per_patch.values())]

    # Classification and Evaluation per patch
    predictions = []
    pvals = []

    for i, input_sample in tqdm(enumerate(input_samples), desc="Classifying Samples"):
        pvalues = calculate_cdfs_pvalues(real_population_cdfs, input_sample)
        pval = combine_pvalues(pvalues, method='fisher', weights=np.ones_like(pvalues) / len(pvalues)).pvalue
        pred = 1 if pval < threshold else 0 
        predictions.append(pred)
        pvals.append(pval)

    # Save heatmaps if required
    if save_independence_heatmaps:
        keys = list(input_sample.keys())
        distributions = np.array(predictions).T
        save_independence_test_heatmaps(keys, distributions)
    
    # Evaluation if test labels are provided
    if test_labels:
        evaluate_predictions(predictions, test_labels)
    
    # Plotting p-values for real and fake labels
    pvals_real = [p for p, label in zip(pvals, test_labels) if label == 0]
    pvals_fake = [p for p, label in zip(pvals, test_labels) if label == 1]

    if save_kdes:
        plot_kdes(np.array(pvals_real), np.array(pvals_fake), f'{num_patches}_patches_wave_{wavelet}_pvals_dist.png', "Stouffer's test: KDE of p-values")


if __name__ == "__main__":
    set_seed(42)
    
    # Directory paths
    data_dir_real = 'data/CelebaHQMaskDataset/train/images_faces' # 'data/real/train'  # Population dataset (unchanged)
    data_dir_fake_real = 'data/CelebaHQMaskDataset/test/images_faces' # 'data/real/test'  # Inference real samples
    data_dir_fake = 'data/stable-diffusion-face-dataset/1024/both_faces' #'data/fake'  # Inference fake samples
    
    # Parameters
    batch_size = 256
    num_samples_per_class = 2449
    pval_threshold = 0.05

    # Define transforms (e.g., resizing and converting to tensor)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    # Create the real population dataset
    real_population_dataset = ImageDataset(data_dir_real, transform=transform, labels=0)

    # Create the inference dataset by sampling from real and fake directories
    inference_data = create_inference_dataset(data_dir_fake_real, data_dir_fake, num_samples_per_class, classes='real')

    # Extract file paths and labels from the sampled data
    image_paths = [x[0] for x in inference_data]
    labels = [x[1] for x in inference_data]

    # Create a custom dataset using ImageDataset with the sampled images and corresponding labels
    inference_dataset = ImageDataset(image_paths, labels, transform=transform)

    # Run the main function with the created datasets
    # for wave in ['bior1.1', 'bior3.1', 'bior6.8', 'coif1', 'coif10', 'db1', 'db38', 'haar', 'rbio6.8', 'sym2']:
    #   for patch in [16, 32, 64, 128]:
    #       main(real_population_dataset, inference_dataset, labels, batch_size=batch_size, wavelet='fourier', threshold=pval_threshold, patch_size=(patch, patch), reload_cdfs=False)
    main(real_population_dataset, inference_dataset, labels, batch_size=batch_size, wavelet='sym2', threshold=pval_threshold, patch_size=(64, 64), reload_cdfs=False)
