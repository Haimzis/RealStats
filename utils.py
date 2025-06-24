from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import random
import warnings
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
import torch
import numpy as np
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from scipy.stats import chi2_contingency
from sklearn.metrics import auc, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve
import os
import seaborn as sns
import networkx as nx
import numpy as np
from scipy.stats import chi2, kstest, gaussian_kde
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from PIL import Image


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


def calculate_chi2_and_corr(i, j, dist_1, dist_2, bins):
    """Compute chi-square p-value and correlation for two distributions."""
    try:
        corr, p_value = spearmanr(dist_1, dist_2)
        # correlation = abs(np.corrcoef(dist_1, dist_2)[0, 1])
        correlation = abs(corr)
        contingency_table, _, _ = np.histogram2d(dist_1, dist_2, bins=(bins, bins))
        chi2_stat, chi2_p, df, expected = chi2_contingency(contingency_table)
        return i, j, chi2_p, correlation
    except ValueError:
        return i, j, -1, correlation


def plot_contingency_table(contingency_table, save_path=None):
    """
    Plots a contingency table as a heatmap with cell values displayed.

    Args:
        contingency_table (numpy.ndarray): The contingency table to plot.
        save_path (str, optional): Path to save the plot. If None, the plot is not saved.

    Returns:
        None
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(contingency_table, cmap='viridis', interpolation='nearest')
    
    # Add cell values
    for i in range(contingency_table.shape[0]):
        for j in range(contingency_table.shape[1]):
            plt.text(j, i, str(np.round(contingency_table[i, j], 2)), ha='center', va='center', color='white', fontsize=10)
    
    # Add labels, title, and colorbar
    plt.colorbar(label='Value')
    plt.title("Contingency Table")
    plt.xlabel("Column")
    plt.ylabel("Row")
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    

def calculate_chi2(i, j, dist_1, dist_2, bins):
    """Compute chi-square p-value and correlation for two distributions."""
    try:
        contingency_table, _, _ = np.histogram2d(dist_1, dist_2, bins=(bins, bins))
        chi2_stat, chi2_p, df, expected = chi2_contingency(contingency_table)
        return i, j, chi2_p
    except ValueError:
        return i, j, -1
    

def compute_chi2_matrix(keys, distributions, max_workers=128, plot_independence_heatmap=False, output_dir='logs', bins=10):
    """Compute Chi-Square p-value matrix and correlation matrix."""
    num_dists = len(distributions)
    chi2_p_matrix = np.zeros((num_dists, num_dists))

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, key1 in enumerate(keys):
            dist_1 = distributions[i]
            for j, key2 in enumerate(keys):
                if i <= j:  # Skip duplicates and diagonal
                    continue
                dist_2 = distributions[j]
                tasks.append(executor.submit(calculate_chi2, i, j, dist_1, dist_2, bins))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing Chi2 and Correlation tests..."):
            i, j, chi2_p = future.result()
            if chi2_p is not None:
                chi2_p_matrix[i, j] = chi2_p
                chi2_p_matrix[j, i] = chi2_p  # Symmetry

    if plot_independence_heatmap:
        create_heatmap(chi2_p_matrix, keys, 'Chi-Square Test (P-values)', output_dir, 'chi2_heatmap.png', annot=len(keys) < 64)

    return chi2_p_matrix, None


def compute_chi2_and_corr_matrix(keys, distributions, max_workers=128, plot_independence_heatmap=False, output_dir='logs', bins=10):
    """Compute Chi-Square p-value matrix and correlation matrix."""
    num_dists = len(distributions)
    chi2_p_matrix = np.zeros((num_dists, num_dists))
    corr_matrix = np.zeros((num_dists, num_dists))

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, key1 in enumerate(keys):
            dist_1 = distributions[i]
            for j, key2 in enumerate(keys):
                if i <= j:  # Skip duplicates and diagonal
                    continue
                dist_2 = distributions[j]
                tasks.append(executor.submit(calculate_chi2_and_corr, i, j, dist_1, dist_2, bins))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing Chi2 and Correlation tests..."):
            i, j, chi2_p, corr = future.result()
            if chi2_p is not None:
                chi2_p_matrix[i, j] = chi2_p
                chi2_p_matrix[j, i] = chi2_p  # Symmetry

            if chi2_p is not None:
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # Symmetry

    if plot_independence_heatmap:
        create_heatmap(chi2_p_matrix, keys, 'Chi-Square Test (P-values)', output_dir, 'chi2_heatmap.png', annot=len(keys) < 64)
        create_heatmap(corr_matrix, keys, 'Correlation Matrix', output_dir, 'corr_heatmap.png', annot=len(keys) < 64)

    return chi2_p_matrix, corr_matrix


def find_largest_independent_group(keys, chi2_p_matrix, p_threshold=0.05):
    """Find the largest independent group using the Chi-Square p-value matrix."""
    G = nx.Graph()
    G.add_nodes_from(keys)
    
    # Add edges where p-values are above the threshold
    indices = np.triu(chi2_p_matrix, k=1) > p_threshold
    rows, cols = np.where(indices)
    edges = np.column_stack((np.array(keys)[rows], np.array(keys)[cols]))
    G.add_edges_from(edges)

    # Subgraph of nodes with edges (dependencies)
    subgraph = G.subgraph([node for node, degree in G.degree() if degree > 0])
    
    # Find the largest independent set (nodes not connected to others)
    independent_set = nx.algorithms.approximation.clique.max_clique(subgraph)
    return list(independent_set) if independent_set else [keys[0]]


def find_largest_independent_group_with_plot(keys, chi2_p_matrix, p_threshold=0.05, output_dir='logs'):
    """Find and plot the largest independent group using the Chi-Square p-value matrix."""
    G = nx.Graph()
    G.add_nodes_from(keys)

    indices = np.triu(chi2_p_matrix, k=1) > p_threshold
    rows, cols = np.where(indices)
    edges = np.column_stack((np.array(keys)[rows], np.array(keys)[cols]))
    G.add_edges_from(edges)

    subgraph = G.subgraph([node for node, degree in G.degree() if degree > 0])
    independent_set = nx.algorithms.approximation.clique.max_clique(subgraph)

    # Plot the graph highlighting the selected clique
    if independent_set:
        os.makedirs(output_dir, exist_ok=True)
        pos = nx.spring_layout(subgraph)
        node_colors = ["tab:red" if n in independent_set else "skyblue" for n in subgraph.nodes()]
        plt.figure(figsize=(10, 8))
        nx.draw(subgraph, pos, with_labels=False, node_color=node_colors, node_size=400)
        nx.draw_networkx_labels(subgraph, pos, font_size=8)
        nx.draw_networkx_edges(subgraph, pos, width=2)
        plt.title("Largest Independent Group")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "independent_graph.png"))
        plt.close()

    return list(independent_set) if independent_set else [keys[0]]


def find_largest_uncorrelated_group(keys, corr_matrix, p_threshold=0.05):
    """Find the largest independent group using the Chi-Square p-value matrix."""
    G = nx.Graph()
    G.add_nodes_from(keys)
    
    # Add edges where p-values are below the threshold
    indices = np.triu(corr_matrix, k=1) < p_threshold
    rows, cols = np.where(indices)
    edges = np.column_stack((np.array(keys)[rows], np.array(keys)[cols]))
    G.add_edges_from(edges)

    # Subgraph of nodes with edges (dependencies)
    subgraph = G.subgraph([node for node, degree in G.degree() if degree > 0])
    
    # Find the largest independent set (nodes not connected to others)
    independent_set = nx.algorithms.approximation.clique.max_clique(subgraph)
    return list(independent_set) if independent_set else [keys[0]]


def find_largest_independent_group_iterative(keys, chi2_p_matrix, p_threshold=0.05):
    """Find the largest independent group using the Chi-Square p-value matrix."""
    G = nx.Graph()
    G.add_nodes_from(keys)
    
    indices = np.triu(chi2_p_matrix, k=1) > p_threshold
    rows, cols = np.where(indices)
    edges = np.column_stack((np.array(keys)[rows], np.array(keys)[cols]))
    G.add_edges_from(edges)

    # Subgraph of nodes with edges (dependencies)
    subgraph = G.subgraph([node for node, degree in G.degree() if degree > 0])
    
    # All maximal cliques for node
    cliques = list(nx.find_cliques(subgraph))
    return cliques


def create_heatmap(data, keys, title, output_dir, filename, figsize=(50, 25), annot=True):
    sorted_indices = np.argsort(keys)
    sorted_keys = [keys[i] for i in sorted_indices]
    sorted_data = data[sorted_indices][:, sorted_indices]


    mask = np.zeros_like(sorted_data, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Apply mask to zero-out the unwanted part
    sorted_data = np.ma.masked_where(mask, sorted_data)
    
    plt.figure(figsize=figsize)
    sns.heatmap(sorted_data, annot=annot, fmt=".2f", cmap=sns.color_palette("YlOrBr", as_cmap=True), 
                xticklabels=sorted_keys, yticklabels=sorted_keys, mask=mask, 
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


def split_population_histogram(real_population_histogram, portion):
    """
    Split the population histogram into tuning and training portions.
    - If portion <= 1.0: Use it as a fraction of the total population.
    - If portion > 1.0: Split into two histograms of sizes `portion` and `N - portion`.
    - If the population size is smaller than the portion, return a single split and None with a warning.
    """

    # Get the population length
    population_length = len(list(real_population_histogram.values())[0])

    # Ensure portion is valid
    if portion <= 0:
        raise ValueError(f"Invalid portion: {portion}. Must be greater than 0.")

    # Handle case where portion is larger than population size
    if (portion <= 1.0 and population_length * portion > population_length) or (portion > 1.0 and portion > population_length):
        warnings.warn(f"Portion {portion} is larger than the population size {population_length}. Returning a single split.")
        return real_population_histogram, None

    # Generate shuffled indices
    indices = list(range(population_length))
    random.shuffle(indices)

    if portion <= 1.0:
        # Portion as a fraction
        split_point = int(population_length * portion)
    else:
        # Portion as an absolute size
        split_point = int(portion)

    # Create the tuning and training histograms based on shuffled indices
    tuning_histogram = {
        k: [v[i] for i in indices[:split_point]] for k, v in real_population_histogram.items()
    }
    training_histogram = {
        k: [v[i] for i in indices[split_point:]] for k, v in real_population_histogram.items()
    }

    return tuning_histogram, training_histogram


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


def plot_cdf(cdf_data, title="Empirical CDF Plot", xlabel="Value", ylabel="CDF", output_file='plot.png'):
    """
    Plot the CDF in a discrete, binned style using the original bins and cumulative values.
    """
    _, bin_edges, cdf = cdf_data

    # Plot the discrete CDF
    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges[:-1], cdf, width=np.diff(bin_edges), align='edge', alpha=0.7, label="Empirical CDF")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.savefig(output_file)


def remove_nans_from_tests(tests_dict):
    """
    Filters out tests from a dictionary where any test contains NaN values.

    Args:
        tests_dict (dict): A dictionary where keys are test names and values are NumPy arrays.

    Returns:
        dict: A new dictionary with tests that do not contain NaN values.
    """
    cleaned_tests = {}

    for test_name, values in tests_dict.items():
        if np.isnan(values).any():
            warnings.warn(f"Test '{test_name}' contains NaN values and will be excluded.")
        else:
            cleaned_tests[test_name] = values

    return cleaned_tests


def compute_dist_cdf(distribution="normal", size=10000, bins=1000):
    """
    Compute and return the histogram, bin edges, and CDF for a given distribution.
    """
    # Generate data based on the specified distribution
    if distribution == "normal":
        data = np.random.normal(loc=0, scale=1, size=size)  # Standard normal
    elif distribution == "uniform":
        data = np.random.uniform(low=0, high=1, size=size)  # Standard uniform

    # Compute the histogram and CDF
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    cdf = np.cumsum(hist) * np.diff(bin_edges)
    
    return hist, bin_edges, cdf


def compute_cdf(histogram_values, bins=1000, test_id=None):
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

        
def plot_pvalue_histograms(real_pvalues, fake_pvalues, figname, title, xlabel="p-values", ylabel="Density"):
    """Plot histograms for real and fake p-values with transparency and save to a file."""
    plt.figure(figsize=(12, 6))
    plt.hist(real_pvalues, bins=100, alpha=0.5, label="Real", color='blue', density=True, edgecolor='k')
    plt.hist(fake_pvalues, bins=100, alpha=0.5, label="Fake", color='orange', density=True,  edgecolor='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def plot_histograms(hist, figname='plot.png', title='histogram', density=True, bins=50):
    """Plot histograms for real and fake p-values with transparency and save to a file."""
    plt.figure(figsize=(12, 6))
    plt.hist(hist, bins=bins, alpha=0.5, color='blue', density=density, edgecolor='k')
    plt.xlabel("p-values")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname)


def plot_binned_histogram(hist, bin_edges, figname='plot.png'):
    """Plot a histogram from precomputed bin counts and bin edges."""
    widths = np.diff(bin_edges)
    plt.bar(bin_edges[:-1], hist, width=widths, align='edge',
            alpha=0.7, edgecolor='k')
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.savefig(figname)


def plot_uniform_and_nonuniform(pvalue_distributions, uniform_indices, output_dir, bins=50):
    """
    Plots 5 uniform and 5 non-uniform p-value distributions in a 2x5 grid.
    Displays the p-value of the KS test for uniformity on each plot.
    """
    # Convert uniform_indices to a set for faster lookup
    uniform_set = set(uniform_indices)
    uniform_distributions = [pvalue_distributions[i] for i in uniform_indices[:5]]

    # Select up to 5 non-uniform distributions
    nonuniform_distributions = [
        pvalue_distributions[i] for i in range(pvalue_distributions.shape[0]) if i not in uniform_set
    ][:5]

    # Create a 2x5 grid for plots
    fig, axes = plt.subplots(2, 5, figsize=(40, 8))

    # Add column titles
    fig.suptitle("Uniform vs Non-Uniform P-Value Distributions", fontsize=16)

    # Plot uniform distributions in the first row
    for idx, ax in enumerate(axes[0, :]):
        if idx < len(uniform_distributions):
            dist = uniform_distributions[idx]
            ax.hist(dist, bins=bins, alpha=0.5, color='blue', density=False, edgecolor='k')
            pval = kstest(dist, 'uniform').pvalue
            ax.text(0.05, 0.95, f"p-value: {pval:.3f}", transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        else:
            ax.axis('off')  # Leave the remaining cells blank

        ax.set_xlabel("p-value")
        ax.set_ylabel("Density")

    # Plot non-uniform distributions in the second row
    for idx, ax in enumerate(axes[1, :]):
        if idx < len(nonuniform_distributions):
            dist = nonuniform_distributions[idx]
            ax.hist(dist, bins=bins, alpha=0.5, color='red', density=False, edgecolor='k')
            pval = kstest(dist, 'uniform').pvalue
            ax.text(0.05, 0.95, f"p-value: {pval:.3f}", transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        else:
            ax.axis('off')  # Leave the remaining cells blank

        ax.set_xlabel("p-value")
        ax.set_ylabel("Density")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to leave space for the title

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "uniform_vs_nonuniform_distributions_2x5.png")
    plt.savefig(output_path)
    plt.close()


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
    return roc_auc


def plot_stouffer_analysis(
    pvalues, inverse_z_scores, stouffer_statistic, stouffer_pvalues, 
    plot_bins=50, output_folder='logs', num_plots_pvalues=10, num_plots_zscores=10
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


def plot_ks_vs_pthreshold(thresholds, ks_pvalues, output_dir):
    """
    Plots KS p-value vs p-threshold.
    """
    sorted_indices = np.argsort(thresholds)
    sorted_thresholds = np.array(thresholds)[sorted_indices]
    sorted_ks_pvalues = np.array(ks_pvalues)[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_thresholds, sorted_ks_pvalues, marker='o', linestyle='-', label="KS p-value")
    plt.xlabel("p_threshold")
    plt.ylabel("KS p-value")
    plt.title("KS p-value vs p-threshold")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "ks_pvalue_wrt_p_threshold.png"))


def AUC_tests_filter(tuning_pvalue_distributions, fake_calibration_pvalue_distributions, auc_threshold=0.6):
    """
    Calculates AUC scores for each test and selects indices with AUC > threshold.
    Adjusts for the relationship where smaller p-values indicate outliers.
    
    Parameters:
    - tuning_pvalue_distributions (np.ndarray): P-value distributions for real data (shape: tests x samples).
    - fake_calibration_pvalue_distributions (np.ndarray): P-value distributions for fake data (shape: tests x samples).
    - auc_threshold (float): Threshold for selecting the best keys based on AUC (default: 0.6).
    
    Returns:
    - auc_scores (np.ndarray): AUC scores for each test.
    - best_keys (np.ndarray): Indices of tests with AUC > auc_threshold.
    """
    # Reverse the p-values
    real_scores = 1 - tuning_pvalue_distributions
    fake_scores = 1 - fake_calibration_pvalue_distributions
    
    # Combine distributions and labels
    combined_pvalues = np.concatenate([real_scores, fake_scores], axis=1)
    combined_labels = np.concatenate([
        np.zeros_like(tuning_pvalue_distributions),
        np.ones_like(fake_calibration_pvalue_distributions)
    ], axis=1)
    
    # Remove entries with NaN values
    # TODO: why are there NaN values? - check the input data
    valid_indices = ~np.isnan(combined_pvalues).any(axis=1)
    combined_pvalues = combined_pvalues[valid_indices]
    combined_labels = combined_labels[valid_indices]

    # Calculate AUC scores using list comprehension
    auc_scores = np.array([roc_auc_score(combined_labels[i], combined_pvalues[i]) for i in range(combined_pvalues.shape[0])])
    
    # Select best keys with AUC > threshold
    best_keys = np.where(auc_scores > auc_threshold)[0]
    
    return auc_scores, best_keys


def save_to_csv(keys, auc_scores, filename="auc_scores.csv"):
    # Create a DataFrame from the arrays
    df = pd.DataFrame({"Keys": keys, "AUC Scores": auc_scores})
    
    # Save DataFrame to a CSV file
    df.to_csv(filename, index=False)
    
    print(f"File saved as {filename}")


def compute_mean_std_dict(input_dict):
    """
    Given a dictionary where each key maps to an array of shape (N, L),
    compute the mean and std along axis=1 and store them in a new dictionary.
    
    Args:
        input_dict (dict): Dictionary with arrays of shape (N, L) where L varies.
    
    Returns:
        dict: Dictionary with mean and std stored as new keys with '_mean' and '_std' suffixes.
    """
    output_dict = {}

    for key, array in input_dict.items():
        # Ensure the array is a NumPy array
        array = np.asarray(array)

        if array.shape[-1] == 1:
            output_dict[key] = array.squeeze()
            continue

        # Compute mean and std along axis 1 (for each row)
        mean_values = np.mean(array, axis=1)  # Shape: (N,)
        std_values = np.std(array, axis=1)    # Shape: (N,)

        # Store in dictionary with updated keys
        output_dict[f"{key}_mean"] = mean_values
        output_dict[f"{key}_std"] = std_values

    return output_dict


def plot_pvalue_histograms_from_arrays(
    real_pvals_array, 
    inference_pvals_array, 
    artifact_path,
    keys
):
    """
    Plots p-value histograms for each test using plot_pvalue_histograms."
    """

    _, T = real_pvals_array.shape
    assert inference_pvals_array.shape[-1] ==  T, "Input arrays must have the same shape"

    for t in range(T):
        real_pvals = real_pvals_array[:, t]
        inf_pvals = inference_pvals_array[:, t]
        corrected_name = keys[t].replace(" ", "_").replace(".", "_").replace("=", "-")
        output_file = f"{artifact_path}_test_{corrected_name}"

        plot_pvalue_histograms(
            real_pvals,
            inf_pvals,
            output_file,
            title=f"Histogram of P-values - Test {corrected_name}"
        )

def build_backbones_statistics_list(models, noise_levels, prefix="RIGID"):
    """
    Generates a list of wave names in the format RIGID.{MODEL}.{NOISE}.

    Args:
        models (list of str): Model names, e.g. ['DINO', 'CLIP']
        noise_levels (list of str): Noise levels as strings, e.g. ['001', '01', '05']
        prefix (str): Optional prefix, default is 'RIGID'

    Returns:
        list of str: All wave combinations like RIGID.DINO.01, RIGID.CLIP.05, ...
    """
    return [f"{prefix}.{model}.{noise}" for model in models for noise in noise_levels]


import os
import numpy as np
import matplotlib.pyplot as plt

def plot_fakeness_score_distribution(results, test_id, output_dir, threshold=0.5):
    """
    Plot fakeness scores per sample with block-wise shuffling,
    styled to match the current matplotlib environment configuration.

    Parameters:
    - results (dict): Must contain 'scores' (fakeness score: high=fake) and 'labels' (0=real, 1=fake).
    - test_id (str): Unique ID for the test (used for filename).
    - output_dir (str): Directory to save the plot.
    - threshold (float): Score threshold for decision boundary (default 0.5).

    Returns:
    - plot_path (str): File path of the saved plot.
    """
    # Extract and shuffle scores per class
    scores = np.array(results['scores'])
    labels = np.array(results['labels'])

    real_scores = np.random.permutation(scores[labels == 0])
    fake_scores = np.random.permutation(scores[labels == 1])

    real_x = np.arange(len(real_scores))
    fake_x = np.arange(len(real_scores), len(real_scores) + len(fake_scores))

    # Plot
    plt.figure(figsize=(12, 4))
    plt.scatter(real_x, real_scores, alpha=0.5, label='Real', s=10, color='tab:blue')
    plt.scatter(fake_x, fake_scores, alpha=0.5, label='Fake', s=10, color='tab:orange')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')

    plt.xlabel("Sample Index")
    plt.ylabel("Fakeness Prob")
    plt.title("Ensemble Probability per Sample (Block-Shuffled)")
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{test_id}_fakeness_score_plot.svg")
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Fakeness score plot saved to: {plot_path}")
    return plot_path


def plot_fakeness_score_histogram(results, test_id, output_dir, threshold=0.5):
    """
    Plot a histogram of fakeness scores for real and fake samples
    with consistent matplotlib styling.

    Parameters:
    - results (dict): Contains 'scores' (fakeness scores) and 'labels' (0=real, 1=fake)
    - test_id (str): Identifier used in the filename
    - output_dir (str): Directory to save the plot
    - threshold (float): Threshold line to show on the plot
    """
    scores = np.array(results['scores'])
    labels = np.array(results['labels'])

    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]

    # Plot
    plt.figure(figsize=(8, 4))
    plt.hist(real_scores, bins=50, alpha=0.6, label='Real', density=True, color='tab:blue')
    plt.hist(fake_scores, bins=50, alpha=0.6, label='Fake', density=True, color='tab:orange')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'significance level α = {1 - threshold:.2f}')

    plt.title("Ensemble Probability Distribution")
    plt.xlabel("Fakeness Prob")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{test_id}_classifier_score_histogram.svg")
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Classifier score histogram saved to: {plot_path}")
    return plot_path


def plot_kde_with_image_markers(pvals_real, image_pvals, image_labels, figsize=(6, 6), bw=0.2):
    """
    Plots a KDE of the real p-value distribution and overlays image p-value markers.
    
    Args:
        pvals_real (array-like): Reference real p-value distribution.
        image_pvals (array-like): List of p-values for individual images.
        image_labels (list of str): List of labels for each image ('real' or 'fake').
        figsize (tuple): Size of the output figure (default: (6, 6)).
        bw (float): Bandwidth for KDE smoothing (default: 0.2).
    """
    # Compute KDE
    kde = gaussian_kde(pvals_real, bw_method=bw)
    x_vals = np.linspace(0, 1, 1000)
    y_vals = kde(x_vals)

    # Create figure
    plt.figure(figsize=figsize)
    plt.plot(x_vals, y_vals, color='blue', linewidth=2)
    plt.fill_between(x_vals, y_vals, color='blue', alpha=0.4)

    # Plot vertical lines for each image
    for pval, label in zip(image_pvals, image_labels):
        color = 'green' if label == 'real' else 'red'
        plt.axvline(pval, ymin=0, ymax=0.3, color=color, linestyle='--', alpha=0.8, linewidth=2)

    # Clean styling
    plt.xlabel("")
    plt.ylabel("")
    plt.yticks([])
    plt.xticks(np.linspace(0, 1, 5))
    plt.title("")
    plt.box(False)
    plt.grid(False)
    plt.tight_layout()
    plt.show()



import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns


def save_per_image_kde_and_images(
    image_paths,
    test_labels,
    tuning_real_population_pvals,
    input_samples_pvalues,
    independent_statistics_keys_group,
    output_dir,
    max_per_class=3
):
    """
    Saves KDE plots with vertical lines marking each selected image's p-value,
    along with a copy of the image.

    Args:
        image_paths (list): List of image file paths.
        test_labels (list): Corresponding list of labels (0 for real, 1 for fake).
        tuning_real_population_pvals (np.ndarray): [num_samples x num_statistics] real p-values.
        input_samples_pvalues (list of list): Per-image p-values for the statistics.
        independent_statistics_keys_group (list): List of statistic keys in order.
        output_dir (str): Directory to save plots.
        max_per_class (int): Max number of images per class to plot.
    """
    save_dir = os.path.join(output_dir, "image_kde_markers")
    os.makedirs(save_dir, exist_ok=True)

    num_stats = len(independent_statistics_keys_group)
    num_real_samples, stats_dim = tuning_real_population_pvals.shape
    num_test_images = len(image_paths)

    # Ensure statistics match
    assert stats_dim == num_stats, f"Expected {num_stats} statistics, but tuning_real_population_pvals has {stats_dim}"
    assert len(test_labels) == len(image_paths) == len(input_samples_pvalues), "Mismatch in number of test images, labels, or p-values"

    for i, pvals in enumerate(input_samples_pvalues):
        assert len(pvals) == num_stats, f"Sample {i} p-value length mismatch: expected {num_stats}, got {len(pvals)}"

    label_indices = {0: [], 1: []}
    for idx, label in enumerate(test_labels):
        if len(label_indices[label]) < max_per_class:
            label_indices[label].append(idx)

    for label, indices in label_indices.items():
        label_name = 'real' if label == 0 else 'fake'
        label_dir = os.path.join(save_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)

        for i in indices:
            pvals = input_samples_pvalues[i]
            img_path = image_paths[i]
            img_basename = os.path.splitext(os.path.basename(img_path))[0]

            # Save original image
            img_save_path = os.path.join(label_dir, f"{img_basename}.png")
            Image.open(img_path).convert("RGB").save(img_save_path)

            for j, stat_key in enumerate(independent_statistics_keys_group):
                fig, ax = plt.subplots(figsize=(4, 4))

                # Plot KDE of reference (real) p-values
                sns.kdeplot(tuning_real_population_pvals[:, j], fill=True, color='blue', bw_adjust=0.5, ax=ax)

                # Vertical marker for current image's p-value
                color = 'green' if label == 0 else 'red'
                ax.axvline(pvals[j], color=color, linestyle='-', linewidth=2)

                # Clean layout
                ax.set_yticks([])
                ax.set_ylabel("")
                ax.set_xlabel("")
                ax.set_title("")
                ax.grid(False)
                ax.set_xlim(0, 1)

                stat_filename = stat_key.replace("=", "-").replace(",", "_").replace(" ", "_")
                plot_path = os.path.join(label_dir, f"{img_basename}_{stat_filename}.png")
                plt.tight_layout()
                plt.savefig(plot_path, bbox_inches="tight", pad_inches=0.1)
                plt.close()


def save_ensembled_pvalue_kde_and_images(
    image_paths,
    test_labels,
    ensembled_pvalues,
    tuning_ensembled_pvalues,
    output_dir,
    max_per_class=3
):
    """
    Saves KDE plots of ensembled p-values for selected real/fake images,
    with vertical lines showing where each image falls.

    Args:
        image_paths (list): List of image file paths.
        test_labels (list): Corresponding labels (0=real, 1=fake).
        ensembled_pvalues (list or np.ndarray): Inference-time ensembled p-values.
        tuning_ensembled_pvalues (np.ndarray): Reference distribution from training set.
        output_dir (str): Directory to save output.
        max_per_class (int): Number of images per class to save.
    """
    save_dir = os.path.join(output_dir, "ensemble_kde_markers")
    os.makedirs(save_dir, exist_ok=True)

    label_indices = {0: [], 1: []}
    for idx, label in enumerate(test_labels):
        if len(label_indices[label]) < max_per_class:
            label_indices[label].append(idx)

    for label, indices in label_indices.items():
        label_name = 'real' if label == 0 else 'fake'
        label_dir = os.path.join(save_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)

        for i in indices:
            img_path = image_paths[i]
            img_basename = os.path.splitext(os.path.basename(img_path))[0]

            # Save original image
            img_save_path = os.path.join(label_dir, f"{img_basename}.png")
            Image.open(img_path).convert("RGB").save(img_save_path)

            # Plot KDE + marker
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.kdeplot(tuning_ensembled_pvalues, fill=True, color='blue', bw_adjust=0.5, ax=ax)

            color = 'green' if label == 0 else 'red'
            ax.axvline(ensembled_pvalues[i], color=color, linestyle='-', linewidth=2)

            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_title("")
            ax.grid(False)
            ax.set_xlim(0, 1)

            plot_path = os.path.join(label_dir, f"{img_basename}_ensemble.png")
            plt.tight_layout()
            plt.savefig(plot_path, bbox_inches="tight", pad_inches=0.1)
            plt.close()