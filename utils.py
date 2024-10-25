import pickle
import random
from matplotlib import pyplot as plt
import torch
import numpy as np
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from scipy.stats import chi2_contingency
from sklearn.metrics import classification_report, confusion_matrix, mutual_info_score
import os
import seaborn as sns


def save_independence_test_heatmaps(keys, distributions, output_dir='logs'):
    num_dists = len(distributions)
    
    # Create matrices to store percentage p-values (pval * 100) for each test
    chi2_p_matrix = np.zeros((num_dists, num_dists))
    mutual_info_p_matrix = np.zeros((num_dists, num_dists))

    # Perform pairwise comparisons
    for i, key in enumerate(keys):
        dist_1 = distributions[i]

        for j, key2 in enumerate(keys):
            if i == j:
                chi2_p_matrix[i, j] = np.nan  # No comparison with itself
                mutual_info_p_matrix[i, j] = np.nan
                continue

            dist_2 = distributions[j]

            ### CHI^2 TEST ###
            bins = np.linspace(0, 1, 21)  # N bins between 0 and 1

            # Bin the p-values into categories for both distributions
            hist_1, _ = np.histogram(dist_1, bins=bins)
            hist_2, _ = np.histogram(dist_2, bins=bins)

            # Create the contingency table by stacking the binned histograms
            contingency_table = np.array([hist_1, hist_2])

            try:
                chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
                chi2_p_matrix[i, j] = chi2_p * 100  # Store p-value * 100
            except ValueError:
                chi2_p_matrix[i, j] = np.nan  # If test fails, set as NaN

            ### MUTUAL INFORMATION TEST ###
            try:
                mutual_info = mutual_info_score(hist_1, hist_2)
                mutual_info_p_matrix[i, j] = mutual_info * 100  # Store MI value * 100 (proxy for p-value)
            except ValueError:
                mutual_info_p_matrix[i, j] = np.nan  # If test fails, set as NaN

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create and save heatmaps for each test
    create_heatmap(chi2_p_matrix, keys, 'Chi-Square Test (P-values %)', output_dir, 'chi2_heatmap.png')
    create_heatmap(mutual_info_p_matrix, keys, 'Mutual Information (P-values %)', output_dir, 'mutual_info_heatmap.png', figsize=(40, 25))


def create_heatmap(data, keys, title, output_dir, filename, figsize=(25, 25)):
    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=True, fmt=".2f", cmap=sns.color_palette("YlOrBr", as_cmap=True), xticklabels=keys, yticklabels=keys, cbar_kws={'label': 'Percentage (%)'}, annot_kws={"size": 10})
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title(title)
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess(image_batch, wavelet_decompose: DTCWTForward, wavelet_compose: DTCWTInverse):
    """Apply wavelet transform and reconstruction to a batch of images"""
    Yl, Yh = wavelet_decompose(image_batch)  # Perform wavelet decomposition
    Yh_zero = [torch.zeros_like(h) for h in Yh]  # Initialize all Yh components to zero
    Yl = torch.zeros_like(Yl)  # Set Yl to zeros
    Yh_zero[0] = Yh[0]  # Retain only the first Yh component
    reconstructed_batch = wavelet_compose((Yl, Yh_zero))  # Reconstruct the image batch
    return reconstructed_batch


def plot_pvalues_vs_bh_threshold(p_values_per_test, alpha=0.05, figname='k_m_plot.png'):
    """
    Plots the p-values vs. their order along with the k/m line (Benjamini-Hochberg threshold).
    """
    # Sort the p-values
    sorted_pvals = np.sort(p_values_per_test)
    m = len(p_values_per_test)

    # Calculate k/m line (BH threshold)
    k = np.arange(1, m+1)
    bh_line = (k * alpha) / m

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(k, sorted_pvals, label="p-values", marker="o", linestyle="--")
    plt.plot(k, bh_line, label="k/m line (BH threshold)", color="red", linestyle="-")

    # Labels and title
    plt.xlabel("Order of p-values")
    plt.ylabel("P-values")
    plt.title("P-values vs. k/m (BH Threshold)")
    plt.legend()

    # Display the plot
    plt.savefig(figname)


def create_inference_dataset(real_dir, fake_dir, num_samples_per_class, classes='both'):
    """Create a balanced dataset for inference by sampling images from real and fake directories."""
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))]

    # Randomly sample from each class
    sampled_real_images = random.sample(real_images, num_samples_per_class)
    sampled_fake_images = random.sample(fake_images, num_samples_per_class)

    # Create dataset of tuples (file_path, label)
    if classes == 'both':
        inference_data = [(img, 0) for img in sampled_real_images] + [(img, 1) for img in sampled_fake_images]
    elif classes == 'fake':
        inference_data = [(img, 1) for img in sampled_fake_images]
    else: # Real
        inference_data = [(img, 0) for img in sampled_real_images] 

    random.shuffle(inference_data)  # Shuffle the dataset
    return inference_data


def evaluate_predictions(predictions, labels):
    """Calculate confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(labels, predictions))


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


def compute_cdfs(histogram_values, bins=10_000):
    """Compute CDFs from histogram values for each wavelet descriptor."""
    cdfs = {}
    for descriptor, values in histogram_values.items():
        hist, bin_edges = np.histogram(values, bins=10_000, density=True)
        cdf = np.cumsum(hist) * np.diff(bin_edges)
        cdfs[descriptor] = (hist, bin_edges, cdf)  # Store both the histogram and the CDF
    return cdfs


def save_population_cdfs(cdfs, file_path):
    """Save the population CDFs to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(cdfs, f)



def load_population_cdfs(file_path):
    """Load the population CDFs from a file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)