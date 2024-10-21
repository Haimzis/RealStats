import random
from matplotlib import pyplot as plt
import torch
import numpy as np
from pytorch_wavelets import DTCWTForward, DTCWTInverse

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
