import numpy as np
from processing.histograms import BaseHistogram


class RGBHistogram(BaseHistogram):
    """Generate a histogram of scalar statistics from a batch of images."""
    def preprocess(self, image_batch):
        """Compute scalar values for the batch and return their histogram."""
        # Move tensor to CPU and convert to NumPy (H, W, C format)
        image_batch = image_batch.permute(0, 2, 3, 1).cpu().numpy()

        # Compute mean and std per channel
        means = np.mean(image_batch, axis=(1, 2))  # Shape: (B, 3)
        stds = np.std(image_batch, axis=(1, 2))    # Shape: (B, 3)

        # Compute scalar values
        scalars = np.sqrt(np.sum(stds**2, axis=1)) + np.sum(means, axis=1) / np.sqrt(3)

        return scalars