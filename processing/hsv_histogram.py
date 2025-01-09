import numpy as np
import cv2
from processing.histograms import BaseHistogram


class HSVHistogram(BaseHistogram):
    """Generate statistics from a specific channel in the HSV color space."""
    def __init__(self, channel: str = "value"):
        """
        Args:
            channel (str): One of ['hue', 'saturation', 'value'].
        """
        super().__init__()
        assert channel in ["hue", "saturation", "value"], "Invalid channel!"
        self.channel = channel

    def preprocess(self, image_batch):
        image_batch = image_batch.permute(0, 2, 3, 1).cpu().numpy()
        channel_idx = {"hue": 0, "saturation": 1, "value": 2}[self.channel]
        statistics = []

        for image in image_batch:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            channel_data = hsv_image[:, :, channel_idx]
            statistics.append(np.std(channel_data) + np.mean(channel_data))

        return np.array(statistics)
