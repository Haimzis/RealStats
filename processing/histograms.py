import torch
import numpy as np
from tqdm import tqdm


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

