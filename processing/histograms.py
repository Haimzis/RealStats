import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BaseHistogram:
    """Abstract base class for generating histograms from transformed images."""

    def __init__(self, max_memory_gb=-1):
        self.device = self._pick_free_gpu(max_memory_gb)

    def _pick_free_gpu(self, max_memory_gb):
        if not torch.cuda.is_available():
            return torch.device("cpu")

        max_bytes = max(max_memory_gb * 1024 ** 3, max_memory_gb)
        best_gpu = None
        best_free_mem = -1

        for i in range(torch.cuda.device_count()):
            free_mem, _ = torch.cuda.mem_get_info(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory
            if max_memory_gb == -1 or free_mem > (total_mem - max_bytes):
                if free_mem > best_free_mem:
                    best_gpu = i
                    best_free_mem = free_mem

        if best_gpu is not None:
            return torch.device(f"cuda:{best_gpu}")
        return torch.device("cpu")

    def preprocess(self, image_batch):
        """Abstract method to apply the desired transformation and calculate the metric."""
        raise NotImplementedError("The 'preprocess' method must be implemented by subclasses.")

    def create_histogram(self, data_loader):
        """Generate histograms for all images in the dataset."""
        results = {}

        for images, _, paths in tqdm(data_loader, desc="Generating histograms", leave=False):
            B, P = images.shape[:2]
            images = images.view(B * P, *images.shape[2:]) # Cross batches
            images = images.to(self.device)
            histograms = self.preprocess(images)
            torch.cuda.empty_cache()
            histograms = histograms.reshape(B, P)

            for sample, path in zip(histograms, paths):
                results[path] = sample

        return results

    def plot_histograms(self, histograms_real, histograms_fake, figname='histogram.png', xlabel='Metric', title='Histogram'):
        """Plot histograms for real and fake datasets with mean and std in a rectangle."""
        # Calculate mean and standard deviation for both datasets
        mean_real = np.mean(histograms_real)
        std_real = np.std(histograms_real)
        mean_fake = np.mean(histograms_fake)
        std_fake = np.std(histograms_fake)

        # Plot histograms
        plt.figure(figsize=(10, 5))
        plt.hist(histograms_real.flatten(), bins=200, alpha=0.5, label="Real", color='blue', density=True)
        plt.hist(histograms_fake.flatten(), bins=200, alpha=0.5, label="Fake", color='red', density=True)

        # Labels and title
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.title(title)
        plt.legend()

        # Add mean and std in a rectangle
        stats_text_real = f"Real: Mean = {mean_real:.2f}, Std = {std_real:.2f}"
        stats_text_fake = f"Fake: Mean = {mean_fake:.2f}, Std = {std_fake:.2f}"

        text_x, text_y = 0.75, 0.85
        plt.gca().add_patch(patches.Rectangle((text_x, text_y), 0.24, 0.1, transform=plt.gca().transAxes,
                                              fill=True, color="white", alpha=0.7, edgecolor="black"))
        plt.text(text_x + 0.01, text_y + 0.05, stats_text_real, transform=plt.gca().transAxes, fontsize=10)
        plt.text(text_x + 0.01, text_y + 0.02, stats_text_fake, transform=plt.gca().transAxes, fontsize=10)

        plt.savefig(figname)
        plt.close()








