import torch
import torch.nn.functional as F

from processing.histograms import BaseHistogram


### 1. Edge Norm Histogram (5×5 Kernel)
class EdgeNormHistogram(BaseHistogram):
    """Detects high-intensity edge structures in latent space using a 5×5 edge-detection kernel."""

    def __init__(self):
        super().__init__()
        self.edge_filter = torch.tensor(
            [[-1, -1, -1, -1, -1],
             [-1,  1,  2,  1, -1],
             [-1,  2,  4,  2, -1],
             [-1,  1,  2,  1, -1],
             [-1, -1, -1, -1, -1]], dtype=torch.float32
        ).view(1, 1, 5, 5)  

    def preprocess(self, image_batch):
        channels = image_batch.shape[1]
        edge_filter = self.edge_filter.to(image_batch.device).repeat(channels, 1, 1, 1)
        edge_response = F.conv2d(image_batch, edge_filter, padding=2, groups=channels)
        return torch.norm(edge_response.view(edge_response.size(0), -1), dim=1).cpu().numpy()


### 2. Smoothing Norm Histogram
class SmoothingNormHistogram(BaseHistogram):
    """Measures image smoothness in latent space using an averaging kernel."""

    def __init__(self):
        super().__init__()
        self.smoothing_filter = torch.ones((1, 1, 5, 5), dtype=torch.float32) / 25  

    def preprocess(self, image_batch):
        channels = image_batch.shape[1]
        smoothing_filter = self.smoothing_filter.to(image_batch.device).repeat(channels, 1, 1, 1)
        smooth_response = F.conv2d(image_batch, smoothing_filter, padding=2, groups=channels)
        return torch.norm(smooth_response.view(smooth_response.size(0), -1), dim=1).cpu().numpy()


### 3. Noise Detection Norm Histogram
class NoiseNormHistogram(BaseHistogram):
    """Extracts noise components from latent space by subtracting smoothed values."""

    def __init__(self):
        super().__init__()
        self.smoothing_filter = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9  

    def preprocess(self, image_batch):
        channels = image_batch.shape[1]
        smoothing_filter = self.smoothing_filter.to(image_batch.device).repeat(channels, 1, 1, 1)
        smoothed = F.conv2d(image_batch, smoothing_filter, padding=1, groups=channels)
        noise_component = image_batch - smoothed
        return torch.norm(noise_component.view(noise_component.size(0), -1), dim=1).cpu().numpy()


### 4. Sharpness Norm Histogram (Laplacian Filter)
class SharpnessNormHistogram(BaseHistogram):
    """Captures image sharpness using a Laplacian filter in latent space."""

    def __init__(self):
        super().__init__()
        self.laplacian_filter = torch.tensor(
            [[ 0, -1,  0],
             [-1,  4, -1],
             [ 0, -1,  0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

    def preprocess(self, image_batch):
        channels = image_batch.shape[1]
        laplacian_filter = self.laplacian_filter.to(image_batch.device).repeat(channels, 1, 1, 1)
        sharpness_response = F.conv2d(image_batch, laplacian_filter, padding=1, groups=channels)
        return torch.norm(sharpness_response.view(sharpness_response.size(0), -1), dim=1).cpu().numpy()


### 5. Emboss Norm Histogram
class EmbossNormHistogram(BaseHistogram):
    """Applies an emboss filter to highlight texture depth in latent space."""

    def __init__(self):
        super().__init__()
        self.emboss_filter = torch.tensor(
            [[-2, -1,  0],
             [-1,  1,  1],
             [ 0,  1,  2]], dtype=torch.float32
        ).view(1, 1, 3, 3)

    def preprocess(self, image_batch):
        channels = image_batch.shape[1]
        emboss_filter = self.emboss_filter.to(image_batch.device).repeat(channels, 1, 1, 1)
        emboss_response = F.conv2d(image_batch, emboss_filter, padding=1, groups=channels)
        return torch.norm(emboss_response.view(emboss_response.size(0), -1), dim=1).cpu().numpy()


### 6. High-Pass Filter Norm Histogram
class HighPassNormHistogram(BaseHistogram):
    """Extracts high-frequency details using a high-pass filter."""

    def __init__(self):
        super().__init__()
        self.highpass_filter = torch.tensor(
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

    def preprocess(self, image_batch):
        channels = image_batch.shape[1]
        highpass_filter = self.highpass_filter.to(image_batch.device).repeat(channels, 1, 1, 1)
        highpass_response = F.conv2d(image_batch, highpass_filter, padding=1, groups=channels)
        return torch.norm(highpass_response.view(highpass_response.size(0), -1), dim=1).cpu().numpy()


### 7. Directional Edge Norm Histogram (Sobel Filters)
class DirectionalEdgeNormHistogram(BaseHistogram):
    """Measures structured edges using horizontal and vertical Sobel filters."""

    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor(
            [[-1,  0,  1],
             [-2,  0,  2],
             [-1,  0,  1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.sobel_y = self.sobel_x.transpose(0, 1)

    def preprocess(self, image_batch):
        channels = image_batch.shape[1]
        sobel_x = self.sobel_x.to(image_batch.device).repeat(channels, 1, 1, 1)
        sobel_y = self.sobel_y.to(image_batch.device).repeat(channels, 1, 1, 1)
        edge_x = F.conv2d(image_batch, sobel_x, padding=1, groups=channels)
        edge_y = F.conv2d(image_batch, sobel_y, padding=1, groups=channels)
        edge_response = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return torch.norm(edge_response.view(edge_response.size(0), -1), dim=1).cpu().numpy()


### 8. Gaussian Difference Norm Histogram
class GaussianDifferenceNormHistogram(BaseHistogram):
    """Computes the difference between two Gaussian blurs for texture analysis."""

    def __init__(self):
        super().__init__()

    def preprocess(self, image_batch):
        blur_1 = F.avg_pool2d(image_batch, 3, stride=1, padding=1)
        blur_2 = F.avg_pool2d(image_batch, 5, stride=1, padding=2)
        diff = blur_1 - blur_2
        return torch.norm(diff.view(diff.size(0), -1), dim=1).cpu().numpy()
