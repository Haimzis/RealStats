import matplotlib.pyplot as plt
import numpy as np

def plot_auc_improvement_bar_chart(generators, auc_before, auc_after, figname="adaptability_improvement.svg", figsize=(8, 5), bar_width=0.35):
    """
    Plot a grouped bar chart showing average AUC scores before and after incorporating ManifoldBias.

    Args:
        generators (list of str): Names of the generators.
        auc_before (list of float): Average AUC scores before ManifoldBias.
        auc_after (list of float): Average AUC scores after ManifoldBias.
        figname (str): Path to save the plot.
        figsize (tuple): Size of the figure.
        bar_width (float): Width of each bar.
    """
    x = np.arange(len(generators))

    plt.figure(figsize=figsize)
    bars1 = plt.bar(x - bar_width/2, auc_before, width=bar_width, label='Before', color='tab:blue')
    bars2 = plt.bar(x + bar_width/2, auc_after, width=bar_width, label='After', color='tab:orange')

    # Annotate bars with values
    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.3f}", ha='center', va='bottom', fontsize=12)

    plt.xticks(x, generators)
    plt.ylabel("Average AUC")
    plt.title("Average AUC Before and After ManifoldBias")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

# --- Example usage ---
generators = ["GauGAN",  "SeeingDark", "Stable Diffusion v2"]
auc_before = [0.822, 0.516, 0.672]
auc_after = [0.949, 0.701, 0.761]

plot_auc_improvement_bar_chart(generators, auc_before, auc_after)
