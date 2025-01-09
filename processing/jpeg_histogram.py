import torch
import numpy as np
import cv2

from processing.histograms import BaseHistogram


class JPEGCompressionRatioHistogram(BaseHistogram):
    """Compute JPEG compression ratio for each image in a batch without saving."""

    def __init__(self, quality=85):
        self.quality = quality  # JPEG quality setting (default: 85)

    def preprocess(self, image_batch):
        """Compute JPEG compression ratio for each image."""
        # Move tensor to CPU and convert to NumPy (H, W, C format)
        image_batch = image_batch.permute(0, 2, 3, 1).cpu().numpy()  # Shape: (B, H, W, C)

        compression_ratios = []
        for img in image_batch:
            # Convert image to uint8
            img = (img * 255).astype(np.uint8)

            # Compress the image in memory using cv2.imencode
            success, jpeg_data = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if not success:
                raise RuntimeError("JPEG compression failed.")

            # Calculate original and compressed sizes
            original_size = img.nbytes  # Uncompressed size in bytes
            compressed_size = len(jpeg_data)  # Compressed size in bytes

            # Compute compression ratio
            compression_ratio = compressed_size / original_size
            compression_ratios.append(compression_ratio)

        return np.array(compression_ratios)
