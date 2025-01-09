import numpy as np
from pytorch_wavelets import DTCWTForward
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class TSNEWavelet:
    """Class for generating t-SNE plots using wavelet-transformed images"""

    def __init__(self, transform=None, J=7, selected_indices=None):
        self.transform = transform
        self.J = J
        self.selected_indices = selected_indices

    def preprocess(self, image_batch):
        """Apply wavelet transform and extract the selected wavelet coefficients"""
        wavelet_decompose = DTCWTForward(J=self.J).to('cuda')
        Yl, Yh = wavelet_decompose(image_batch)

        if self.selected_indices:
            real_Yh = [Yh[i][..., 0] for i in self.selected_indices if i < len(Yh)]
        else:
            real_Yh = [h[..., 0] for h in Yh]

        return real_Yh

    def extract_wavelets_for_tsne(self, data_loader):
        """Extract wavelet features from all images for t-SNE"""
        all_wavelets = []
        labels = []
        
        for images, lbls in tqdm(data_loader, desc="Extracting wavelet features", leave=False):
            images = images.to('cuda')
            wavelet_features = self.preprocess(images)
            
            flattened_wavelets = np.concatenate([h.view(h.size(0), -1).cpu().numpy() for h in wavelet_features], axis=1)
            all_wavelets.append(flattened_wavelets)
            labels.extend(lbls.cpu().numpy())
        
        return np.vstack(all_wavelets), np.array(labels)

    def run_tsne(self, real_loader, fake_loader):
        """Run t-SNE on wavelet features extracted from real and fake datasets"""
        wavelets_real, labels_real = self.extract_wavelets_for_tsne(real_loader)
        wavelets_fake, labels_fake = self.extract_wavelets_for_tsne(fake_loader)
        
        # Combine real and fake data for t-SNE
        X = np.vstack([wavelets_real, wavelets_fake])
        y = np.hstack([labels_real, labels_fake])
        
        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(X)

        # Plot t-SNE results
        self.plot_tsne_results(tsne_results, y)

    def plot_tsne_results(self, tsne_results, labels):
        """Plot t-SNE results and label the real and fake points"""
        plt.figure(figsize=(8, 6))
        real_points = tsne_results[labels == 0]
        plt.scatter(real_points[:, 0], real_points[:, 1], label='Real', alpha=0.5, color='blue')
        fake_points = tsne_results[labels == 1]
        plt.scatter(fake_points[:, 0], fake_points[:, 1], label='Fake', alpha=0.5, color='red')
        plt.title("t-SNE on Wavelet Features")
        plt.legend()
        plt.savefig("tsne.png")
