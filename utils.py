import pickle
import random
from matplotlib import pyplot as plt
from sklearn.isotonic import spearmanr
import torch
import numpy as np
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from scipy.stats import chi2_contingency
from sklearn.metrics import auc, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve
import os
import seaborn as sns
import networkx as nx
import numpy as np
from scipy.stats import chi2


def view_subgraph(subgraph, title="Subgraph Visualization", save_path='subgraph.png'):
    """
    Visualize the subgraph using networkx and matplotlib.
    
    Parameters:
    -----------
    subgraph : networkx.Graph
        The subgraph to be visualized.
    title : str
        Title of the plot.
    save_path : str or None
        Path to save the plot as an image. If None, display interactively.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(subgraph)  # Layout for positioning the nodes
    
    # Assign random colors to edges
    edges = subgraph.edges()
    edge_colors = ["#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)]) for _ in edges]
        
    nx.draw(subgraph, pos, with_labels=False, node_size=400, node_color="skyblue", font_size=10, font_weight="bold")
    nx.draw_networkx_edges(subgraph, pos, edgelist=edges, edge_color=edge_colors, width=2)

    plt.title(title)
    plt.savefig(save_path, format='PNG')
    print(f"Subgraph saved to {save_path}")


def chi_square_independence_test(observed):
    """
    chi2_stat : float
        Chi-square statistic.
    df : int
        Degrees of freedom.
    p_value : float
        P-value of the test.
    expected : 2D array
        Expected counts under null hypothesis of independence.
    """
    # Compute row and column sums
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    grand_total = observed.sum()

    # Compute expected counts
    expected = np.outer(row_totals, col_totals) / grand_total

    # Compute the chi-square statistic
    chi2_stat = ((observed - expected)**2 / (expected + np.finfo(np.float16).eps)).sum()

    # Degrees of freedom
    r, c = observed.shape
    df = (r - 1) * (c - 1)

    # Compute p-value
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return chi2_stat, p_value, df, expected


def get_largest_independent_subgroup(keys, distributions, p_threshold=0.05, plot_independence_heatmap=False, output_dir='logs'):
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
    G = nx.Graph()
    G.add_nodes_from(keys)

    num_dists = len(distributions)
    chi2_p_matrix = np.zeros((num_dists, num_dists))
    corr_p_matrix = np.zeros((num_dists, num_dists))

    sorted_keys = sorted(keys)
    pvalues = []

    for i, key1 in enumerate(sorted_keys):
        dist_1 = distributions[i]
        for j, key2 in enumerate(sorted_keys):
            if i <= j:
                continue
            
            dist_2 = distributions[j]

            contingency_table, _, _ = np.histogram2d(dist_1, dist_2, bins=(401, 401))
            # _chi2_stat, _p_val, _df, _expected = chi_square_independence_test(contingency_table)

            # TODO: adding HP fine tuning for finding best threshold for CHI2, to get normality for stouffer.
            try:
                chi2_stat, chi2_p, df, expected = chi2_contingency(contingency_table)
                _p_val = chi2_p 
                chi2_p_matrix[i, j] = _p_val
                corr_p_matrix[i, j] = np.corrcoef(dist_1, dist_2)[0, 1]

                pvalues.append(_p_val)
                # Add an edge if the pair is independent (high p-value)
                if _p_val > p_threshold:
                    G.add_edge(key1, key2)
            except ValueError:
                pass

    if plot_independence_heatmap:
        create_heatmap(chi2_p_matrix, sorted_keys, 'Chi-Square Test (P-values %)', output_dir, 'chi2_heatmap.png')
        create_heatmap(corr_p_matrix, sorted_keys, 'Spearman Correlation Test', output_dir, 'corr_heatmap.png')

    subgraph = G.subgraph([node for node, degree in G.degree() if degree > 0])
    independent_set = nx.algorithms.approximation.clique.max_clique(subgraph)
    largest_independent_group = list(independent_set)
    largest_independent_group = [sorted_keys[0]] if len(largest_independent_group) == 0 else largest_independent_group
    return largest_independent_group


def create_heatmap(data, keys, title, output_dir, filename, figsize=(50, 25)):
    mask = np.zeros_like(data, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Apply mask to zero-out the unwanted part
    data = np.ma.masked_where(mask, data)
    
    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=True, fmt=".2f", cmap=sns.color_palette("YlOrBr", as_cmap=True), 
                xticklabels=keys, yticklabels=keys, mask=mask, 
                cbar_kws={'label': 'Percentage (%)'}, annot_kws={"size": 10})
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
        hist, bin_edges = np.histogram(values, bins=bins, density=False)
        pdf = hist / np.sum(hist)
        cdf = np.cumsum(pdf)
        # cdf = np.cumsum(hist) * np.diff(bin_edges) # Density=True
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

        
def plot_pvalue_histograms(real_pvalues, fake_pvalues, figname, title):
    """Plot histograms for real and fake p-values with transparency and save to a file."""
    plt.figure(figsize=(12, 6))
    plt.hist(real_pvalues, bins=100, alpha=0.5, label="Real", color='blue', density=True, edgecolor='k')
    plt.hist(fake_pvalues, bins=100, alpha=0.5, label="Fake", color='orange', density=True,  edgecolor='k')
    plt.xlabel("p-values")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname)


def plot_histograms(hist, figname='plot.png', title='histogram'):
    """Plot histograms for real and fake p-values with transparency and save to a file."""
    plt.figure(figsize=(12, 6))
    plt.hist(hist, bins=100, alpha=0.5, color='blue', density=True, edgecolor='k')
    plt.xlabel("p-values")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname)


def plot_roc_curve(results, test_id, output_dir):
    """Plot ROC curve for a specific test."""
    labels = results['labels']
    scores = results['scores']
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curves
    plt.figure()
    plt.plot(fpr, tpr, label=f"{results['n_tests']} Tests (AUC = {roc_auc:.2f})")

    plt.title(f'ROC Curve for Test: {test_id}')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'roc_curve.png'))
    plt.close()


def plot_stouffer_analysis(
    pvalues, inverse_z_scores, stouffer_statistic, stouffer_pvalues, 
    plot_bins=100, output_folder='logs', num_plots_pvalues=10, num_plots_zscores=10
):
    """
    Generate and save plots for the p-values, inverse z-scores, Stouffer's statistics, and Stouffer's p-values.

    Parameters:
    - pvalues: Array of p-values (N x T).
    - inverse_z_scores: Array of inverse z-scores (N x T).
    - stouffer_statistic: Array of combined z-scores (N x 1).
    - stouffer_pvalues: Array of Stouffer's combined p-values (N x 1).
    - plot_bins: Number of bins for the plotted histograms.
    - output_folder: Folder to save the output figures.
    - num_plots_pvalues: Number of columns to display for p-values subplots.
    - num_plots_zscores: Number of columns to display for inverse z-scores subplots.
    """
    T = pvalues.shape[1]  # Number of tests

    # Step 1: Plot sampled p-values for the first `num_plots_pvalues` tests
    fig_pvalues, axes_pvalues = plt.subplots(1, min(num_plots_pvalues, T), figsize=(20, 4))
    if T == 1:  # If there's only one test, axes_pvalues is not iterable
        axes_pvalues = [axes_pvalues]

    for i in range(min(num_plots_pvalues, T)):
        axes_pvalues[i].hist(pvalues[:, i], bins=plot_bins, edgecolor='k', alpha=0.7, density=True, color='blue')
        axes_pvalues[i].set_title(f"P-Values Test {i+1}")
        axes_pvalues[i].set_xlabel("P-Value")
        axes_pvalues[i].set_ylabel("Density")

    fig_pvalues.tight_layout()
    fig_pvalues.savefig(f"{output_folder}/pvalues_histogram_subset.png")
    plt.close(fig_pvalues)

    # Step 2: Plot inverse z-scores for the first `num_plots_zscores` tests
    fig_zscores, axes_zscores = plt.subplots(1, min(num_plots_zscores, T), figsize=(20, 4))
    if T == 1:  # If there's only one test, axes_zscores is not iterable
        axes_zscores = [axes_zscores]

    for i in range(min(num_plots_zscores, T)):
        axes_zscores[i].hist(inverse_z_scores[:, i], bins=plot_bins, edgecolor='k', alpha=0.7, density=True, color='orange')
        axes_zscores[i].set_title(f"Z-Scores Test {i+1}")
        axes_zscores[i].set_xlabel("Z-Score Value")
        axes_zscores[i].set_ylabel("Density")

    fig_zscores.tight_layout()
    fig_zscores.savefig(f"{output_folder}/inverse_zscores_histogram_subset.png")
    plt.close(fig_zscores)

    # Step 3: Plot Stouffer's statistics
    fig_stouffer_stat, ax_stouffer_stat = plt.subplots(figsize=(10, 6))
    ax_stouffer_stat.hist(stouffer_statistic, bins=plot_bins, edgecolor='k', alpha=0.7, density=True, color='green')
    ax_stouffer_stat.set_title("Histogram of Stouffer's Statistics")
    ax_stouffer_stat.set_xlabel("Stouffer Statistic")
    ax_stouffer_stat.set_ylabel("Density")
    fig_stouffer_stat.tight_layout()
    fig_stouffer_stat.savefig(f"{output_folder}/stouffer_statistics_histogram.png")
    plt.close(fig_stouffer_stat)

    # Step 4: Plot Stouffer's p-values
    fig_stouffer_pvalues, ax_stouffer_pvalues = plt.subplots(figsize=(10, 6))
    ax_stouffer_pvalues.hist(stouffer_pvalues, bins=plot_bins, edgecolor='k', alpha=0.7, density=True, color='purple')
    ax_stouffer_pvalues.set_title("Histogram of Stouffer's P-Values")
    ax_stouffer_pvalues.set_xlabel("P-Value")
    ax_stouffer_pvalues.set_ylabel("Density")
    fig_stouffer_pvalues.tight_layout()
    fig_stouffer_pvalues.savefig(f"{output_folder}/stouffer_pvalues_histogram.png")
    plt.close(fig_stouffer_pvalues)
