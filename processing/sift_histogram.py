import numpy as np
import cv2
from processing.histograms import BaseHistogram

class SIFTHistogram(BaseHistogram):
    """
    Generate unique histograms using the SIFT feature detector
    and descriptor provided by OpenCV, processing images one-by-one
    in a simple loop.
    """

    def __init__(self, grid_size=8, num_bins=8):
        """
        Initialize SIFT histogram generator.
        Args:
            grid_size: The size of the grid cells (in pixels).
            num_bins: Number of bins in the orientation histogram.
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_bins = num_bins
        self.sift = cv2.SIFT_create()  # Use OpenCV's SIFT implementation

    def preprocess(self, image_batch):
        """
        Compute SIFT-based histograms of orientations for a batch of images.
        Args:
            image_batch: A torch tensor of shape (B, C, H, W).
        Returns:
            A numpy array of shape (B,) with the histogram norm per image.
        """
        # Move tensor to CPU and convert to NumPy (H, W, C format)
        image_batch = image_batch.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)

        histograms = []
        for i in range(image_batch.shape[0]):
            image = image_batch[i]

            # Convert from RGB to grayscale if necessary
            if image.shape[-1] == 3:
                gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                # If already single-channel
                gray_image = (image.squeeze() * 255).astype(np.uint8)

            # Detect keypoints and compute SIFT descriptors
            keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)

            # If no descriptors are found, set the sum to 0
            if descriptors is None:
                histograms.append(0.0)
                continue

            # Compute histogram of orientations (0..360) for descriptors
            hist, _ = np.histogram(descriptors.ravel(), bins=self.num_bins, range=(0, 360))

            # Weighted sum is the L2 norm of the histogram
            weighted_sum = np.linalg.norm(hist)
            histograms.append(weighted_sum)

        return np.array(histograms)
