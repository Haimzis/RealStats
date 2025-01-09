import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr

from processing.histograms import BaseHistogram


class PSNRBlurHistogram(BaseHistogram):
    """Compute PSNR for each image in the batch compared to its smoothed version."""

    def __init__(self, max_pixel_value=255.0):
        self.max_pixel_value = max_pixel_value  # Maximum possible pixel value (e.g., 255 for 8-bit images)

    def preprocess(self, image_batch):
        """
        Compute PSNR for each image compared to its Gaussian-blurred version.
        Args:
            image_batch: Torch tensor of images (B, C, H, W).
        Returns:
            psnr_values: NumPy array of PSNR values for each image.
        """
        # Move tensor to CPU and convert to NumPy (H, W, C format)
        image_batch = image_batch.permute(0, 2, 3, 1).cpu().numpy()  # Shape: (B, H, W, C)

        psnr_values = []
        for img in image_batch:
            # Convert to uint8
            img = (img * 255).astype(np.uint8)

            # Convert to grayscale for simplicity
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Create a smoothed version of the image (Gaussian blur)
            blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

            # Compute PSNR using skimage's built-in function
            score = 1- psnr(gray_img, blurred_img, data_range=self.max_pixel_value)
            psnr_values.append(score)

        return np.array(psnr_values)
