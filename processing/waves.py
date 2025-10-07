from pytorch_wavelets import DWTForward
from torch.fft import fft2
from processing.histograms import BaseHistogram


import numpy as np
from scipy.fft import dct


class DCTNormHistogram(BaseHistogram):
    """Generate norm histograms using DCT-transformed images."""

    def __init__(self, dct_type=2, norm='ortho'):
        super().__init__()
        self.dct_type = dct_type  # DCT type (1, 2, 3, or 4)
        self.norm = norm          # Normalization method

    def preprocess(self, image_batch):
        """Apply 2D DCT using SciPy and calculate the L2 norm."""
        # Move tensor to CPU and convert to numpy
        image_batch = image_batch.cpu().numpy()

        # Apply DCT twice (2D DCT) along the last two dimensions (height, width)
        dct_transformed = np.stack([
            dct(dct(image, type=self.dct_type, norm=self.norm, axis=1),
                type=self.dct_type, norm=self.norm, axis=2)
            for image in image_batch
        ])

        # Flatten and compute L2 norm
        flattened = dct_transformed.reshape(dct_transformed.shape[0], -1)
        norm_histogram = np.linalg.norm(flattened, axis=1)

        return norm_histogram


class WaveletEnergyHistogram(BaseHistogram):
    """Generate energy histograms from wavelet-transformed images."""

    def __init__(self, J=7, wave='db1', selected_indices=None):
        super().__init__()
        self.J = J  # Number of wavelet levels
        self.wave = wave
        self.selected_indices = selected_indices  # List of selected Yh indices (sub-bands)

    def preprocess(self, image_batch):
        """Apply wavelet transform and calculate the energy of selected sub-bands."""
        wavelet_decompose = DWTForward(J=self.J, wave=self.wave).to('cuda')
        Yl, Yh = wavelet_decompose(image_batch)

        # Select sub-bands if specified
        if self.selected_indices:
            Yh = [Yh[i] for i in self.selected_indices if i < len(Yh)]

        # Calculate energy for each sub-band
        energies = [torch.sum(h**2, dim=tuple(range(1, h.dim()))).cpu().numpy() for h in Yh]

        # Average energy across selected sub-bands
        energy_histogram = np.mean(energies, axis=0)

        return energy_histogram


class FourierNormHistogram(BaseHistogram):
    """Generate norm histograms using Fourier-transformed images."""

    def preprocess(self, image_batch):
        """Apply Fourier transform and calculate the L2 norm."""
        image_batch = image_batch.to('cuda')
        fourier_transformed = fft2(image_batch)

        # Flatten the transformed images
        flattened = fourier_transformed.view(fourier_transformed.size(0), -1)

        # Calculate L2 norm
        norm_histogram = torch.norm(flattened, dim=1).cpu().numpy()

        return norm_histogram


class WaveletNormHistogram(BaseHistogram):
    """Generate norm histograms from wavelet-transformed images."""

    def __init__(self, J=7, wave='db1', selected_indices=None):
        super().__init__()
        self.J = J  # Number of wavelet levels
        self.wave = wave
        self.selected_indices = selected_indices  # List of selected Yh indices (sub-bands)

    def preprocess(self, image_batch):
        """Apply wavelet transform and calculate the norm of selected sub-bands."""
        wavelet_decompose = DWTForward(J=self.J, wave=self.wave).to('cuda')
        Yl, Yh = wavelet_decompose(image_batch)

        # Select sub-bands if specified
        if self.selected_indices:
            Yh = [Yh[i] for i in self.selected_indices if i < len(Yh)]

        # Flatten and concatenate all sub-bands
        concatenated = torch.cat([h.view(h.size(0), -1) for h in Yh], dim=1)

        # Calculate L2 norm
        norm_histogram = torch.norm(concatenated, dim=1).cpu().numpy()

        return norm_histogram