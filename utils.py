import random
from matplotlib import pyplot as plt
import torch
import numpy as np
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
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



# Set random seed for reproducibility
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


def plot_pvalues_vs_bh_threshold(p_values_per_test, alpha, figname='k_m_plot.png'):
    """
    Plots the p-values vs. their order along with the k/m line (Benjamini-Hochberg threshold).

    Args:
        p_values_per_test (np.array): Array of p-values to plot.
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
