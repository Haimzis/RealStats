import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from processing.histograms import BaseHistogram


class SIFTHistogram(BaseHistogram):
    """
    Generate unique histograms using the SIFT feature detector 
    and descriptor provided by OpenCV.
    """

    def __init__(self, grid_size=8, num_bins=8, max_workers=4):
        """
        Initialize SIFT histogram generator.
        Args:
            grid_size: The size of the grid cells (in pixels).
            num_bins: Number of bins in the orientation histogram.
            max_workers: Maximum number of threads for parallel processing.
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_bins = num_bins
        self.sift = cv2.SIFT_create()  # Use OpenCV's SIFT implementation
        self.max_workers = max_workers

    def _process_single_image(self, image):
        """Process a single image to compute the weighted histogram."""
        if image.shape[-1] == 3:  # Convert RGB to grayscale if necessary
            gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:  # If already grayscale
            gray_image = (image.squeeze() * 255).astype(np.uint8)

        # Detect keypoints and compute SIFT descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)

        # If no descriptors are found, return zero
        if descriptors is None:
            return 0

        # Compute histogram of orientation for descriptors
        hist, _ = np.histogram(
            descriptors.ravel(), bins=self.num_bins, range=(0, 360)
        )

        weighted_sum = np.linalg.norm(hist)
        return weighted_sum


    def preprocess(self, image_batch):
        """
        Compute histograms of orientations using SIFT over grid cells.
        Args:
            image_batch: A batch of images as a torch tensor (B, C, H, W).
        Returns:
            histograms: A numpy array of shape (B,) where each value is the weighted sum of orientation histograms.
        """
        # Move tensor to CPU and convert to numpy (H, W, C format)
        image_batch = image_batch.permute(0, 2, 3, 1).cpu().numpy()

        # Use ThreadPoolExecutor to process images in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            histograms = list(executor.map(self._process_single_image, image_batch))

        return np.array(histograms)
