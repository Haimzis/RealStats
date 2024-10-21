from pytorch_wavelets import DTCWTForward, DWTForward
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pywt

class EnergyHistogram:
    """Class for generating energy histograms from wavelet-transformed images"""

    def __init__(self, transform=None, J=7, selected_indices=None):
        self.transform = transform
        self.J = J  # Number of wavelet levels
        self.selected_indices = selected_indices  # List of selected Yh indices (sub-bands)

    def preprocess(self, image_batch):
        """Apply wavelet transform and calculate the energy of selected sub-bands"""
        # wavelet_decompose = DTCWTForward(J=self.J).to('cuda')
        wavelet_decompose = DWTForward(J=self.J).to('cuda')

        Yl, Yh = wavelet_decompose(image_batch)

        if self.selected_indices:
            real_Yh = [Yh[i][..., 0] for i in self.selected_indices if i < len(Yh)]
        else:
            real_Yh = [h[..., 0] for h in Yh]

        energy_histogram = np.mean([torch.sum(h**2, dim=tuple(range(1, 5))).cpu().numpy() / h.shape[2] for h in real_Yh], axis=0)

        return energy_histogram

    def create_histogram(self, data_loader):
        """Create energy histograms for all images in the dataset"""
        all_histograms = []
        for images, _ in tqdm(data_loader, desc="Generating histograms", leave=False):
            images = images.to('cuda')
            energy_histograms = self.preprocess(images)
            all_histograms.append(energy_histograms)
        
        return np.concatenate(all_histograms, axis=0)

    def plot_histograms(self, histograms_real, histograms_fake):
        """Plot histograms for real and fake datasets"""
        plt.figure(figsize=(10, 5))
        plt.hist(histograms_real.flatten(), bins=200, alpha=0.5, label="Real", color='blue', density=True)
        plt.hist(histograms_fake.flatten(), bins=200, alpha=0.5, label="Fake", color='red', density=True)
        plt.xlabel("Energy")
        plt.ylabel("Frequency")
        plt.title("Energy Histograms for Real and Fake Datasets")
        plt.legend()
        plt.savefig("energy_histogram.png")

class FourierEnergyHistogram:
    def __init__(self, transform=None):
        self.transform = transform

    def preprocess(self, image_batch):
        image_batch = image_batch.to('cuda')
        fourier_transformed = torch.fft.fft2(image_batch)
        energy_histogram = torch.sum(torch.abs(fourier_transformed)**2, dim=(1, 2, 3)).cpu().numpy()
        return energy_histogram

    def create_histogram(self, data_loader):
        all_histograms = []
        for images, _ in tqdm(data_loader, desc="Generating Fourier histograms", leave=False):
            energy_histograms = self.preprocess(images)
            all_histograms.append(energy_histograms)
        return np.concatenate(all_histograms, axis=0)

    def plot_histograms(self, histograms_real, histograms_fake):
        plt.figure(figsize=(10, 5))
        plt.hist(histograms_real.flatten(), bins=200, alpha=0.5, label="Real", color='blue', density=True)
        plt.hist(histograms_fake.flatten(), bins=200, alpha=0.5, label="Fake", color='red', density=True)
        plt.xlabel("Energy")
        plt.ylabel("Frequency")
        plt.title("Energy Histograms using Fourier Transform")
        plt.legend()
        plt.savefig("fourier_energy_histogram.png")


class NormHistogram:
    """Class for generating norm histograms from wavelet-transformed images"""

    def __init__(self, transform=None, J=7, wave='db1', selected_indices=None):
        self.transform = transform
        self.J = J  # Number of wavelet levels
        self.selected_indices = selected_indices  # List of selected Yh indices (sub-bands)
        self.wave = wave

    def preprocess(self, image_batch):
        """Apply wavelet transform and calculate the norm of selected sub-bands"""
        # wavelet_decompose = DTCWTForward(J=self.J).to('cuda')
        wavelet_decompose = DWTForward(J=self.J, wave=self.wave).to('cuda')
        Yl, Yh = wavelet_decompose(image_batch)

        if type(wavelet_decompose) is DTCWTForward:
            if self.selected_indices:
                real_Yh = [Yh[i][..., 0] for i in self.selected_indices if i < len(Yh)]
            else:
                real_Yh = [h[..., 0] for h in Yh]

        elif type(wavelet_decompose) is DWTForward:
            if self.selected_indices:
                real_Yh = [Yh[i] for i in self.selected_indices if i < len(Yh)]    
        
        norm_histogram = np.linalg.norm(np.concatenate([h.view(h.size(0), -1).cpu().numpy() for h in real_Yh], axis=1), axis=1)

        return norm_histogram

    def create_histogram(self, data_loader):
        """Create norm histograms for all images in the dataset"""
        all_histograms = []
        for images, _ in tqdm(data_loader, desc="Generating histograms", leave=False):
            images = images.to('cuda')
            norm_histograms = self.preprocess(images)
            all_histograms.append(norm_histograms)
        
        return np.concatenate(all_histograms, axis=0)

    def plot_histograms(self, histograms_real, histograms_fake, figname='norm_histogram.png'):
        """Plot histograms for real and fake datasets with mean and std in a rectangle"""
        
        # Calculate mean and standard deviation for both datasets
        mean_real = np.mean(histograms_real)
        std_real = np.std(histograms_real)
        
        mean_fake = np.mean(histograms_fake)
        std_fake = np.std(histograms_fake)
        
        # Plot histograms
        plt.figure(figsize=(10, 5))
        plt.hist(histograms_real.flatten(), bins=200, alpha=0.5, label="Real", color='blue', density=True)
        plt.hist(histograms_fake.flatten(), bins=200, alpha=0.5, label="Fake", color='red', density=True)
        
        # Labels and title
        plt.xlabel("L2 Norm")
        plt.ylabel("Frequency")
        plt.title("L2 Norm Histograms for Real and Fake Datasets")
        
        # Add legend
        plt.legend()

        # Create text for mean and std
        stats_text_real = f"Real: Mean = {mean_real:.2f}, Std = {std_real:.2f}"
        stats_text_fake = f"Fake: Mean = {mean_fake:.2f}, Std = {std_fake:.2f}"
        
        # Position for the text box (adjust the coordinates as needed)
        text_x = 0.75
        text_y = 0.85
        
        # Add the rectangle
        plt.gca().add_patch(patches.Rectangle((text_x, text_y), 0.2, 0.1, transform=plt.gca().transAxes,
                                            fill=True, color="white", alpha=0.7, edgecolor="black"))
        
        # Add the mean and std text inside the rectangle
        plt.text(text_x + 0.01, text_y + 0.05, stats_text_real, transform=plt.gca().transAxes, fontsize=10)
        plt.text(text_x + 0.01, text_y + 0.02, stats_text_fake, transform=plt.gca().transAxes, fontsize=10)
        
        # Save the figure
        plt.savefig(figname)
