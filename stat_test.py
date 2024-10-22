from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
import numpy as np
from tqdm import tqdm
from data_utils import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from statsmodels.stats.multitest import multipletests
from processing.histograms import NormHistogram
from utils import save_independence_test_heatmaps, set_seed
from sklearn.metrics import confusion_matrix, classification_report


def preprocess_wavelet_level(dataset, batch_size, wavelet, wavelet_level):
    """Preprocess the dataset for a single wavelet level and wavelet type using NormHistogram."""
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
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


def compute_cdfs(histogram_values):
    """Compute CDFs from histogram values for each wavelet descriptor."""
    cdfs = {}
    for descriptor, values in histogram_values.items():
        hist, bin_edges = np.histogram(values, bins='auto', density=True)
        cdf = np.cumsum(hist) * np.diff(bin_edges)
        cdfs[descriptor] = (hist, bin_edges, cdf)  # Store both the histogram and the CDF
    return cdfs


def calculate_p_value_for_sample(sample, population_cdf_info, alternative='less'):
    """
    Calculate the p-value for a sample using the precomputed CDF from the population.
    """
    hist, bin_edges, population_cdf = population_cdf_info

    # Find the corresponding bin of the sample
    sample_bin_index = np.digitize(sample, bin_edges) - 1
    sample_bin_index = np.clip(sample_bin_index, 0, len(population_cdf) - 1)  # Ensure index stays within bounds

    # Get CDF value for the sample
    sample_cdf = population_cdf[sample_bin_index]

    if alternative == 'less':
        return sample_cdf
    elif alternative == 'greater':
        return 1 - sample_cdf
    elif alternative == 'both':
        less_p_value = sample_cdf
        greater_p_value = 1 - sample_cdf
        return 2 * min(less_p_value, greater_p_value)
    else:
        raise ValueError("Invalid alternative hypothesis. Choose from 'less', 'greater', or 'both'.")


def perform_fdr_classification(real_population_cdfs, input_samples_values, threshold=0.05):
    """Perform KS test and apply FDR correction using precomputed CDFs."""
    p_values_per_test = []

    for wavelet_descriptor in sorted(real_population_cdfs.keys()):
        # Retrieve precomputed CDF for the population
        population_cdf_info = real_population_cdfs[wavelet_descriptor]

        # Calculate p-value for the input sample using precomputed CDF
        sample = input_samples_values[wavelet_descriptor]
        p_value = calculate_p_value_for_sample(sample, population_cdf_info)
        p_values_per_test.append(p_value)

    ## Apply FDR correction across all tests
    # _, pvals_corrected, _, _ = multipletests(p_values_per_test, method='fdr_bh', alpha=threshold)
    # return int(np.any(pvals_corrected < threshold))  # Return True if any corrected p-value is below the threshold
    
    return p_values_per_test

def create_inference_dataset(real_dir, fake_dir, num_samples_per_class):
    """Create a balanced dataset for inference by sampling images from real and fake directories."""
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))]

    # Randomly sample from each class
    sampled_real_images = random.sample(real_images, num_samples_per_class)
    sampled_fake_images = random.sample(fake_images, num_samples_per_class)

    # Create dataset of tuples (file_path, label)
    # inference_data = [(img, 0) for img in sampled_real_images] + [(img, 1) for img in sampled_fake_images]
    # inference_data = [(img, 1) for img in sachi2_contingencympled_fake_images] # Fake
    inference_data = [(img, 0) for img in sampled_real_images] # Real

    random.shuffle(inference_data)  # Shuffle the dataset

    return inference_data


def evaluate_predictions(predictions, labels):
    """Calculate confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(labels, predictions))


def main(real_population_dataset, inference_dataset, test_labels=None, batch_size=128, threshold=0.05):
    # Wavelet levels to process
    wavelet_levels = [0, 1, 2, 3]
    
    # List of wavelets to process
    wavelet_list = ['bior1.1', 'bior3.1', 'bior6.8', 'coif1', 'coif10', 'db1', 'db38', 'haar', 'rbio6.8', 'sym2']

    # Preprocess real population dataset
    real_population_values_per_wave = parallel_preprocess(real_population_dataset, batch_size, wavelet_list, wavelet_levels)

    # Compute CDFs for real population values
    real_population_cdfs = compute_cdfs(real_population_values_per_wave)

    # Preprocess input samples (inference dataset)
    input_samples_values_per_wave = parallel_preprocess(inference_dataset, batch_size, wavelet_list, wavelet_levels)

    # Prepare for parallelized classification
    input_samples = [dict(zip(input_samples_values_per_wave.keys(), values)) for values in zip(*input_samples_values_per_wave.values())]

    # Perform FDR classification
    predictions = []
    for i, input_sample in tqdm(enumerate(input_samples), desc="Classifying Samples"):
        predictions.append(perform_fdr_classification(real_population_cdfs, input_sample, threshold=threshold))

    keys = list(input_samples[0].keys())
    distributions = np.array(predictions).T

    save_independence_test_heatmaps(keys, distributions)
    
    test_labels=None
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
    batch_size = 1024
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
    main(real_population_dataset, inference_dataset, labels, batch_size=batch_size, threshold=pval_threshold)
