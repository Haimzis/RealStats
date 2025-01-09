import numpy as np
import cv2
from processing.histograms import BaseHistogram

class GaborFilterFeatureMapNorm(BaseHistogram):
    """Generate the norm of feature maps produced by Gabor filters."""
    def preprocess(self, image_batch):
        image_batch = image_batch.permute(0, 2, 3, 1).cpu().numpy()
        statistics = []

        # Parameters for Gabor filter
        kernels = [cv2.getGaborKernel((21, 21), sigma, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
                   for sigma in [1, 3]
                   for theta in np.linspace(0, np.pi, 4)]

        for image in image_batch:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            feature_norm = 0

            for kernel in kernels:
                filtered = cv2.filter2D(gray_image, cv2.CV_64F, kernel)
                feature_norm += np.linalg.norm(filtered)

            statistics.append(feature_norm)

        return np.array(statistics)
