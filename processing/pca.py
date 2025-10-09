from pytorch_wavelets import DTCWTForward
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm

class PCAOutlierDetector:
    """Class for detecting fake samples as outliers using PCA based on real samples"""

    def __init__(self, transform=None, J=7, selected_indices=None):
        self.transform = transform
        self.J = J
        self.selected_indices = selected_indices
        self.pca_model = None  # To store the fitted PCA model

    def preprocess(self, image_batch):
        """Apply wavelet transform and extract the selected wavelet coefficients"""
        wavelet_decompose = DTCWTForward(J=self.J).to('cuda')
        Yl, Yh = wavelet_decompose(image_batch)

        if self.selected_indices:
            real_Yh = [Yh[i][..., 0] for i in self.selected_indices if i < len(Yh)]
        else:
            real_Yh = [h[..., 0] for h in Yh]

        return real_Yh

    def extract_wavelets(self, data_loader):
        """Extract wavelet features from all images for PCA"""
        all_wavelets = []
        for images, _ in tqdm(data_loader, desc="Extracting wavelet features", leave=False):
            images = images.to('cuda')
            wavelet_features = self.preprocess(images)
            flattened_wavelets = np.concatenate([h.view(h.size(0), -1).cpu().numpy() for h in wavelet_features], axis=1)
            all_wavelets.append(flattened_wavelets)
        
        return np.vstack(all_wavelets)

    def run_pca_outlier_detection(self, real_loader, fake_loader):
        """Fit PCA on real data and detect outliers in fake data"""
        wavelets_real = self.extract_wavelets(real_loader)
        wavelets_fake = self.extract_wavelets(fake_loader)

        self.pca_model = PCA(n_components=2)
        pca_results_real = self.pca_model.fit_transform(wavelets_real)

        pca_results_fake = self.pca_model.transform(wavelets_fake)

        return pca_results_real, pca_results_fake
