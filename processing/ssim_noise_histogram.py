import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

from processing.histograms import BaseHistogram


class SSIMBlurHistogram(BaseHistogram):
    """Compute SSIM against a blurred version of each image in the batch."""

    def __init__(self, blur_kernel=(5, 5), sigma=1.0):
        self.blur_kernel = blur_kernel
        self.sigma = sigma

    def preprocess(self, image_batch):
        """Compute SSIM for each image against its blurred version."""
        # Move tensor to CPU and convert to NumPy (H, W, C format)
        image_batch = image_batch.permute(0, 2, 3, 1).cpu().numpy()  # Shape: (B, H, W, C)

        ssim_values = []
        for img in image_batch:
            # Convert to grayscale
            gray_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

            # Apply Gaussian blur
            blurred_img = cv2.GaussianBlur(gray_img, ksize=self.blur_kernel, sigmaX=self.sigma)

            # Compute SSIM
            score = 1 - ssim(gray_img, blurred_img, data_range=gray_img.max() - gray_img.min())
            ssim_values.append(score)

        return np.array(ssim_values)
