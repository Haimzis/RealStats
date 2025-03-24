# statistics_factory.py

from typing import Optional

from processing.blurness_histogram import BlurDetectionHistogram
from processing.filters_histograms import DirectionalEdgeNormHistogram, EdgeNormHistogram, EmbossNormHistogram, GaussianDifferenceNormHistogram, HighPassNormHistogram, NoiseNormHistogram, SharpnessNormHistogram, SmoothingNormHistogram
from processing.gabor_histogram import GaborFilterFeatureMapNorm
from processing.histograms import DCTNormHistogram, FourierNormHistogram, WaveletNormHistogram
from processing.hsv_histogram import HSVHistogram
from processing.jpeg_histogram import JPEGCompressionRatioHistogram
from processing.laplacian_histogram import LaplacianVarianceHistogram
from processing.manifold_bias_histogram import LatentNoiseCriterion
from processing.psnr_noise_histogram import PSNRBlurHistogram
from processing.rigid_histogram import RIGIDHistogram
from processing.sift_histogram import SIFTHistogram
from processing.ssim_noise_histogram import SSIMBlurHistogram

# Import all histogram classes


# Define a dictionary that maps wavelet types to their corresponding histogram generator classes
WAVELET_HISTOGRAMS = {
    # Wavelet and Frequency Domain Statistics
    'bior1.1': lambda level: WaveletNormHistogram(selected_indices=[level], wave='bior1.1'),
    'bior3.1': lambda level: WaveletNormHistogram(selected_indices=[level], wave='bior3.1'),
    'bior6.8': lambda level: WaveletNormHistogram(selected_indices=[level], wave='bior6.8'),
    'coif1': lambda level: WaveletNormHistogram(selected_indices=[level], wave='coif1'),
    'coif10': lambda level: WaveletNormHistogram(selected_indices=[level], wave='coif10'),
    'db1': lambda level: WaveletNormHistogram(selected_indices=[level], wave='db1'),
    'db38': lambda level: WaveletNormHistogram(selected_indices=[level], wave='db38'),
    'haar': lambda level: WaveletNormHistogram(selected_indices=[level], wave='haar'),
    'rbio6.8': lambda level: WaveletNormHistogram(selected_indices=[level], wave='rbio6.8'),
    'sym2': lambda level: WaveletNormHistogram(selected_indices=[level], wave='sym2'),

    'fourier': lambda level: FourierNormHistogram() if level == 0 else None,
    'dct': lambda level: DCTNormHistogram(dct_type=level) if 0 < level <= 4 else None,

    # Non-Wavelet Image Statistics
    'blurness': lambda level: BlurDetectionHistogram() if level == 0 else None,
    'gabor': lambda level: GaborFilterFeatureMapNorm() if level == 0 else None,
    'hsv': lambda level: HSVHistogram() if level == 0 else None,
    'jpeg': lambda level: JPEGCompressionRatioHistogram() if level == 0 else None,
    'laplacian': lambda level: LaplacianVarianceHistogram() if level == 0 else None,
    'psnr': lambda level: PSNRBlurHistogram() if level == 0 else None,
    'sift': lambda level: SIFTHistogram() if level == 0 else None,
    'ssim': lambda level: SSIMBlurHistogram() if level == 0 else None,

    # New Latent Space Statistics
    'edge5x5': lambda level: EdgeNormHistogram() if level == 0 else None,
    'smoothing': lambda level: SmoothingNormHistogram() if level == 0 else None,
    'noise': lambda level: NoiseNormHistogram() if level == 0 else None,
    'sharpness': lambda level: SharpnessNormHistogram() if level == 0 else None,
    'emboss': lambda level: EmbossNormHistogram() if level == 0 else None,
    'highpass': lambda level: HighPassNormHistogram() if level == 0 else None,
    'sobel': lambda level: DirectionalEdgeNormHistogram() if level == 0 else None,
    'gauss_diff': lambda level: GaussianDifferenceNormHistogram() if level == 0 else None,

    # Paper Statistics
    'LatentNoiseCriterion': lambda level: LatentNoiseCriterion() if level == 0 else None,

    # ==============================
    #    Hardcoded RIGID Statistics
    # ==============================
    'RIGID.05': lambda level: RIGIDHistogram(model_name="facebook/dinov2-large", noise_level=0.05) if level == 0 else None,
    'RIGID.10': lambda level: RIGIDHistogram(model_name="facebook/dinov2-large", noise_level=0.10) if level == 0 else None,
    'RIGID.20': lambda level: RIGIDHistogram(model_name="facebook/dinov2-large", noise_level=0.20) if level == 0 else None,
    'RIGID.30': lambda level: RIGIDHistogram(model_name="facebook/dinov2-large", noise_level=0.30) if level == 0 else None,
    'RIGID.50': lambda level: RIGIDHistogram(model_name="facebook/dinov2-large", noise_level=0.50) if level == 0 else None,
}


def get_histogram_generator(wavelet: str, wavelet_level: int) -> Optional[object]:
    """Returns the appropriate histogram generator based on wavelet type and level."""
    return WAVELET_HISTOGRAMS.get(wavelet, lambda level: None)(wavelet_level)
