from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
from data_utils import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from statsmodels.stats.multitest import multipletests
from processing.histograms import NormHistogram
from utils import calculate_p_value_for_sample, compute_cdfs, create_inference_dataset, evaluate_predictions, load_population_cdfs, plot_pvalues_vs_bh_threshold, save_independence_test_heatmaps, save_population_cdfs, set_seed
from scipy.stats import norm
import os


def preprocess_wavelet_level(dataset, batch_size, wavelet, wavelet_level):
    """Preprocess the dataset for a single wavelet level and wavelet type using NormHistogram."""
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    histogram_generator = NormHistogram(selected_indices=[wavelet_level], wave=wavelet)
    histograms = histogram_generator.create_histogram(data_loader)
    return histograms


def parallel_preprocess(dataset, batch_size, wavelet_list, wavelet_levels):
    """Preprocess the dataset for multiple wavelet levels and wavelet types in parallel."""
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_level_wave = {
            executor.submit(preprocess_wavelet_level, dataset, batch_size, wave, level): (wave, level)
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
    

def stouffers_test(p_values, weights=None):
    """
    Performs Stouffer's test to combine p-values from multiple independent tests.
    """
    # Step 1: Convert p-values to Z-scores
    z_scores = norm.ppf(1 - p_values)

    # Step 2: If no weights are provided, assign equal weights (uniform weighting)
    if weights is None:
        weights = np.ones_like(p_values) / len(p_values)

    # Step 3: Apply weighted Stouffer's method
    Z_combined = np.sum(weights * z_scores) / np.sqrt(np.sum(weights ** 2))

    # Step 4: Convert the combined Z-score back to a p-value
    p_combined = 1 - norm.cdf(Z_combined)

    return p_combined


def main(real_population_dataset, inference_dataset, test_labels=None, batch_size=128, threshold=0.05, save_independence_heatmaps=False, cdf_file="population_cdfs_10kb.pkl", reload_cdfs=False):
    # Wavelet levels to process
    wavelet_levels = [0, 1, 2, 3]
    # wavelet_levels = [0]

    # List of wavelets to process
    wavelet_list = ['bior1.1', 'bior3.1', 'bior6.8', 'coif1', 'coif10', 'db1', 'db38', 'haar', 'rbio6.8', 'sym2']
    # subgroup1 = ['db1', 'haar', 'bior1.1']
    # subgroup2 = ['rbio6.8', 'coif10', 'sym2']
    # wavelet_list = subgroup1

    selected_keys = [(wave, level) for wave in wavelet_list for level in wavelet_levels]
    if reload_cdfs and os.path.exists(cdf_file):
        print(f"Loading precomputed population CDFs from {cdf_file}")
        loaded_real_population_cdfs = load_population_cdfs(cdf_file)
        real_population_cdfs = {key: loaded_real_population_cdfs[key] for key in selected_keys if key in loaded_real_population_cdfs}
    else: 
        # Preprocess real population dataset
        real_population_values_per_wave = parallel_preprocess(real_population_dataset, batch_size, wavelet_list, wavelet_levels)

        # Compute CDFs for real population values
        real_population_cdfs = compute_cdfs(real_population_values_per_wave)

        print(f"Saving population CDFs to {cdf_file}")
        save_population_cdfs(real_population_cdfs, cdf_file)
        
    # Preprocess input samples (inference dataset)
    input_samples_values_per_wave = parallel_preprocess(inference_dataset, batch_size, wavelet_list, wavelet_levels)

    # Prepare for parallelized classification
    input_samples = [dict(zip(input_samples_values_per_wave.keys(), values)) for values in zip(*input_samples_values_per_wave.values())]

    # FDR classification
    predictions = []
    for i, input_sample in tqdm(enumerate(input_samples), desc="Classifying Samples"):
        pvalues = calculate_cdfs_pvalues(real_population_cdfs, input_sample)

        # pred = stouffers_test(pvalues)
        # pred = fdr_classification(pvalues, threshold=threshold)

        # predictions.append(pred)
        plot_pvalues_vs_bh_threshold(pvalues, figname=f"k_m_plot{i}.png")

    if save_independence_heatmaps:
        keys = list(input_samples[0].keys())
        distributions = np.array(predictions).T
        save_independence_test_heatmaps(keys, distributions)
    
    # Evaluation
    if test_labels:
        evaluate_predictions(predictions, test_labels)


if __name__ == "__main__":
    set_seed(42)
    
    # Directory paths
    data_dir_real = 'data/real/train'  # Population dataset (unchanged)
    data_dir_fake_real = 'data/real/test'  # Inference real samples
    data_dir_fake = 'data/fake'  # Inference fake samples
    
    # Parameters
    batch_size = 256
    num_samples_per_class = 2449
    pval_threshold = 0.15

    # Define transforms (e.g., resizing and converting to tensor)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    # Create the real population dataset
    real_population_dataset = ImageDataset(data_dir_real, transform=transform, labels=0)

    # Create the inference dataset by sampling from real and fake directories
    inference_data = create_inference_dataset(data_dir_fake_real, data_dir_fake, num_samples_per_class)

    # Extract file paths and labels from the sampled data
    image_paths = [x[0] for x in inference_data]
    labels = [x[1] for x in inference_data]

    # Create a custom dataset using ImageDataset with the sampled images and corresponding labels
    inference_dataset = ImageDataset(image_paths, labels, transform=transform)

    # Run the main function with the created datasets
    main(real_population_dataset, inference_dataset, labels, batch_size=batch_size, threshold=pval_threshold, reload_cdfs=True)
