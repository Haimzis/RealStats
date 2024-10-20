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
from utils import set_seed
from sklearn.metrics import confusion_matrix, classification_report


def preprocess_wavelet_level(dataset, batch_size, wavelet, wavelet_level):
    """Preprocess the dataset for a single wavelet level and wavelet type using NormHistogram."""
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    histogram_generator = NormHistogram(selected_indices=[wavelet_level], wave=wavelet)
    histograms = histogram_generator.create_histogram(data_loader)
    return histograms


def parallel_preprocess(dataset, batch_size, wavelet_list, wavelet_levels):
    """Preprocess the dataset for multiple wavelet levels and wavelet types in parallel."""
    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit tasks for all combinations of wavelet levels and wavelet types
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


def perform_fdr_classification(real_population_values, input_samples_values, threshold=0.05):
    """Perform KS test and apply FDR correction across all wavelet."""
    p_values_per_test = []

    for wavelet_descriptor in sorted(real_population_values.keys()):
        t_stat, p_value = histogram_cdf_test(real_population_values[wavelet_descriptor], input_samples_values[wavelet_descriptor], alternative='less')
        p_values_per_test.append(p_value)

    _, pvals_corrected, _, _ = multipletests(p_values_per_test, method='fdr_bh', alpha=threshold)
    return int(np.any(pvals_corrected < threshold))  # Return True if any corrected p-value is below 0.05
    # return int(np.any(np.array(p_values_per_test) < threshold))  # Return True if any corrected p-value is below 0.05


def create_inference_dataset(real_dir, fake_dir, num_samples_per_class):
    """Create a balanced dataset for inference by sampling images from real and fake directories."""
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))]

    # Randomly sample from each class
    sampled_real_images = random.sample(real_images, num_samples_per_class)
    sampled_fake_images = random.sample(fake_images, num_samples_per_class)

    # Create dataset of tuples (file_path, label)
    inference_data = [(img, 0) for img in sampled_real_images] + [(img, 1) for img in sampled_fake_images]
    random.shuffle(inference_data)  # Shuffle the dataset

    return inference_data


def evaluate_predictions(predictions, labels):
    """Calculate confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(labels, predictions))


def histogram_cdf_test(population, sample, alternative='two-sided'):
    """Perform a test by calculating the CDF from the population histogram and comparing the sample."""
    
    # Compute histogram for the population
    hist, bin_edges = np.histogram(population, bins='auto', density=True)
    
    # Compute CDF from histogram
    cdf = np.cumsum(hist) * np.diff(bin_edges)
    
    # Find the corresponding bin of the sample
    sample_bin_index = np.digitize(sample, bin_edges) - 1
    sample_bin_index = np.clip(sample_bin_index, 0, len(cdf) - 1)  # Ensure index stays within bounds
    
    # Get CDF value for the sample
    sample_cdf = cdf[sample_bin_index]
    
    # Depending on the alternative hypothesis, calculate p-value
    if alternative == 'greater':
        p_value = 1 - sample_cdf  # Higher values in CDF imply lower p-values
    elif alternative == 'less':
        p_value = sample_cdf  # CDF directly gives the p-value for lower comparison
    elif alternative == 'two-sided':
        p_value = 2 * min(sample_cdf, 1 - sample_cdf)  # Two-sided uses both tails
    else:
        raise ValueError("Alternative must be 'greater', 'less', or 'two-sided'")
    
    return sample_cdf, p_value


def main(real_population_dataset, inference_dataset, test_labels=None, batch_size=128, threshold=0.05):
    # Wavelet levels to process
    wavelet_levels = [0, 1, 2, 3]
    
    # List of wavelets to process
    wavelet_list = ['bior1.1', 'bior3.1', 'bior6.8', 'coif1', 'coif10', 'db1', 'db38', 'haar', 'rbio6.8', 'sym2']

    # Preprocess real population dataset
    real_population_values_per_wave = parallel_preprocess(real_population_dataset, batch_size, wavelet_list, wavelet_levels)

    # Preprocess input samples (inference dataset)
    input_samples_values_per_wave = parallel_preprocess(inference_dataset, batch_size, wavelet_list, wavelet_levels)

    # Prepare for parallelized classification
    input_samples = [dict(zip(input_samples_values_per_wave.keys(), values)) for values in zip(*input_samples_values_per_wave.values())]

    # Perform FDR classification
    predictions = []
    for i, input_sample in tqdm(enumerate(input_samples), desc="Classifying Samples"):
        predictions.append(perform_fdr_classification(real_population_values_per_wave, input_sample, threshold=threshold))

    # Evaluation
    if test_labels:
        evaluate_predictions(predictions, test_labels)


if __name__ == "__main__":
    set_seed(42)
    
    # Directory paths
    data_dir_real = 'data/debug'  # Population dataset (unchanged)
    data_dir_fake_real = 'data/real/test'  # Inference real samples
    data_dir_fake = 'data/fake'  # Inference fake samples
    
    # Parameters
    batch_size = 32
    num_samples_per_class = 25
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
