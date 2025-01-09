import numpy as np
import cv2

from processing.histograms import BaseHistogram


class LaplacianVarianceHistogram(BaseHistogram):
    """Compute the variance of the Laplacian for each image in a batch."""

    def __init__(self):
        pass

    def preprocess(self, image_batch):
        """Compute the variance of the Laplacian for each image and return the values."""
        # Move tensor to CPU and convert to NumPy (H, W, C format)
        image_batch = image_batch.permute(0, 2, 3, 1).cpu().numpy()  # Shape: (B, H, W, C)

        # Convert each image to grayscale using cv2
        gray_images = np.array([
            cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            for img in image_batch
        ])  # Shape: (B, H, W)

        # Compute the Laplacian for each image and calculate its variance
        laplacian_variances = np.array([
            cv2.Laplacian(gray_img, cv2.CV_64F).var()
            for gray_img in gray_images
        ])  # Shape: (B,)

        return laplacian_variances
