import pickle
import random
from matplotlib import pyplot as plt
import torch
import numpy as np
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from scipy.stats import chi2_contingency
from sklearn.metrics import auc, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, mutual_info_score, roc_curve
import os
import seaborn as sns
import networkx as nx



def get_largest_independent_subgroup(keys, distributions, p_threshold=0.05, v_threshold=0.1):
    """
    Find the largest sub-group of independent keys.

    Parameters:
    -----------
    keys : list of str
        Names of the distributions.
    distributions : dict
        Dictionary where keys are distribution names and values are 1D arrays of values.
    threshold : float
        The p-value threshold above which two distributions are considered independent.

    Returns:
    --------
    largest_independent_group : list of str
        List of keys representing the largest independent subgroup.
    """
    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(keys)

    # Add edges for dependent pairs
    for i, key1 in enumerate(keys):
        dist_1 = distributions[i]
        for j, key2 in enumerate(keys):
            if i >= j:  # Avoid redundant comparisons and self-comparison
                continue

            dist_2 = distributions[j]

            # Bin the distributions into histograms
            bins = np.linspace(0, 1, 101)
            hist_1, _ = np.histogram(dist_1, bins=bins)
            hist_2, _ = np.histogram(dist_2, bins=bins)

            # Create the contingency table
            contingency_table = np.array([hist_1, hist_2])

            # Perform Chi-Square Test
            try:
                chi2_stat, chi2_p, _, expected = chi2_contingency(contingency_table)

                # Calculate Cramér's V
                n = np.sum(contingency_table)  # Total observations
                min_dim = min(contingency_table.shape) - 1  # Min(rows - 1, cols - 1)
                cramer_v = np.sqrt(chi2_stat / (n * min_dim))

                # Add an edge if the pair is independent (high p-value AND low effect size)
                if chi2_p > p_threshold and cramer_v < v_threshold:
                    G.add_edge(key1, key2)
            except ValueError:
                pass

    # Process the connected subgraph
    subgraph = G.subgraph([node for node, degree in G.degree() if degree > 0])
    independent_set = nx.algorithms.approximation.clique.max_clique(subgraph)
    
    # Combine results
    largest_independent_group =  list(independent_set)
    return largest_independent_group


def save_independence_test_heatmaps(keys, distributions, output_dir='logs'):
    num_dists = len(distributions)
    
    # Create matrices to store percentage p-values (pval * 100) for each test
    chi2_p_matrix = np.zeros((num_dists, num_dists))
    mutual_info_p_matrix = np.zeros((num_dists, num_dists))

    # Perform pairwise comparisons
    for i, key in enumerate(keys):
        dist_1 = distributions[key]

        for j, key2 in enumerate(keys):
            if i == j:
                chi2_p_matrix[i, j] = np.nan  # No comparison with itself
                mutual_info_p_matrix[i, j] = np.nan
                continue

            dist_2 = distributions[key2]

            ### CHI^2 TEST ###
            bins = np.linspace(0, 1, 101)  # N bins between 0 and 1

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
    pass

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
        hist, bin_edges = np.histogram(values, bins=bins, density=True)
        cdf = np.cumsum(hist) * np.diff(bin_edges)
        cdfs[descriptor] = (hist, bin_edges, cdf)  # Store both the histogram and the CDF
    return cdfs

def compute_cdf(histogram_values, bins=10_000):
    """Compute CDFs from histogram values for each wavelet descriptor."""
    hist, bin_edges = np.histogram(histogram_values, bins=bins, density=True)
    cdf = np.cumsum(hist) * np.diff(bin_edges)
    cdf_dict = (hist, bin_edges, cdf)  # Store both the histogram and the CDF
    return cdf_dict


def save_population_histograms(histograms, file_path):
    """Save the population histograms to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(histograms, f)


def load_population_histograms(file_path):
    """Load the population histograms from a file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    

def calculate_metrics(test_labels, predictions):
    """
    Calculate evaluation metrics including precision, recall, specificity, F1-score, and accuracy.
    """
    # Confusion matrix components
    cm = confusion_matrix(test_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist()  # Include CM for reference
    }


def plot_sensitivity_specificity_by_patch_size(results, wavelet, threshold, output_dir):
    """Plot sensitivity (recall) and specificity across wavelets and patches."""
    patches = sorted(results.keys())
    recalls = [results[patch]['recall'] for patch in patches]
    specificities = [results[patch]['specificity'] for patch in patches]

    # Plot recall and specificity
    plt.figure()
    plt.plot(patches, recalls, marker='o', label='Recall (Sensitivity)')
    plt.plot(patches, specificities, marker='x', label='Specificity')
    plt.title(f'Sensitivity and Specificity for Wavelet: {wavelet} (Alpha={threshold})')
    plt.xlabel('Patch Size')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'sensitivity_specificity_{wavelet}_alpha_{threshold}.png'))
    plt.close()

        
def plot_kdes(hist1, hist2, figname, title):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(hist1, label="Real", fill=True, bw_adjust=0.5, gridsize=2000)
    sns.kdeplot(hist2, label="Fake", fill=True, bw_adjust=0.5, gridsize=2000)
    plt.xlabel("p-values")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname)


def plot_sensitivity_specificity_by_num_waves(results, threshold, output_dir):
    """Plot sensitivity (recall) and specificity across number of wavelet tests."""
    num_waves = sorted(results.keys())
    recalls = [results[nw]['recall'] for nw in num_waves]
    specificities = [results[nw]['specificity'] for nw in num_waves]

    # Plot recall and specificity
    plt.figure()
    plt.plot(num_waves, recalls, marker='o', label='Recall (Sensitivity)')
    plt.plot(num_waves, specificities, marker='x', label='Specificity')
    plt.title(f'Sensitivity and Specificity by Number of Wave Tests (Alpha={threshold})')
    plt.xlabel('Number of Wave Tests')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'sensitivity_specificity_by_{num_waves}_num_waves_alpha_{threshold}.png'))
    plt.close()


def plot_roc_curve_by_patch_size(results, wavelet, output_dir):
    """Plot ROC curve across patch sizes for a specific wavelet."""
    patches = sorted(results.keys())
    tprs = []
    fprs = []
    roc_aucs = []

    for patch in patches:
        labels = results[patch]['labels']
        scores = results[patch]['scores']
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)

    # Plot ROC curves
    plt.figure()
    for patch, fpr, tpr, roc_auc in zip(patches, fprs, tprs, roc_aucs):
        plt.plot(fpr, tpr, label=f"Patch {patch}, {results[patch]['n_tests']} Tests (AUC = {roc_auc:.2f})")

    plt.title(f'ROC Curve for Wavelet: {wavelet}')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'roc_curve_{wavelet}_patches.png'))
    plt.close()


def plot_roc_curve_by_num_waves(results, output_dir):
    """Plot ROC curve across number of wavelet tests."""
    num_waves = sorted(results.keys())
    tprs = []
    fprs = []
    roc_aucs = []

    for nw in num_waves:
        labels = results[nw]['labels']
        scores = results[nw]['scores']
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)

    # Plot ROC curves
    plt.figure()
    for nw, fpr, tpr, roc_auc in zip(num_waves, fprs, tprs, roc_aucs):
        plt.plot(fpr, tpr, label=f"{nw} Wavelets, {results[nw]['n_tests']} Tests (AUC = {roc_auc:.2f})")

    plt.title('ROC Curve by Number of Wave Tests')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'roc_curve_by_num_waves.png'))
    plt.close()


