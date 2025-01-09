import numpy as np
import cv2
from processing.histograms import BaseHistogram


class BlurDetectionHistogram(BaseHistogram):
    """Generate a histogram of blur metrics using the Tenengrad method."""
    def preprocess(self, image_batch):
        image_batch = image_batch.permute(0, 2, 3, 1).cpu().numpy()
        statistics = []

        for image in image_batch:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            tenengrad = np.sqrt(sobel_x**2 + sobel_y**2).mean()
            statistics.append(tenengrad)

        return np.array(statistics)
