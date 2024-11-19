from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
# from RBMpaper.python.pipeline import RBMPipeline
from data_utils import PatchDataset
from torch.utils.data import DataLoader
from processing.histograms import FourierNormHistogram, WaveletNormHistogram
from utils import (
    calculate_p_value_for_sample,
    compute_cdf,
    calculate_metrics,
    load_population_cdfs,
    save_population_cdfs,
    plot_kdes,
    save_independence_test_heatmaps,
)
from statsmodels.stats.multitest import multipletests
from scipy.stats import combine_pvalues
import os


def preprocess_wavelet(dataset, batch_size, wavelet, wavelet_level, num_data_workers):
    """Preprocess the dataset for a single wavelet level and wavelet type using NormHistogram."""
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_workers)
    if wavelet in ['bior1.1', 'bior3.1', 'bior6.8', 'coif1', 'coif10', 'db1', 'db38', 'haar', 'rbio6.8', 'sym2']:
        histogram_generator = WaveletNormHistogram(selected_indices=[wavelet_level], wave=wavelet)
    elif wavelet == 'fourier':
        histogram_generator = FourierNormHistogram()
    else:
        raise ValueError('Invalid wavelet type.')
    return histogram_generator.create_histogram(data_loader)


def wave_parallel_preprocess(dataset, batch_size, wavelet_list, wavelet_levels, max_workers, num_data_workers):
    """Preprocess the dataset for multiple wavelet levels and wavelet types in parallel."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_level_wave = {
            executor.submit(preprocess_wavelet, dataset, batch_size, wave, level, num_data_workers): (wave, level)
            for wave in wavelet_list
            for level in wavelet_levels
        }

        for future in as_completed(future_to_level_wave):
            wave, level = future_to_level_wave[future]
            try:
                results[(wave, level)] = future.result()
            except Exception as exc:
                print(f"Wavelet {wave}, Level {level}, generated an exception: {exc}")

    return results


def patch_parallel_preprocess(original_dataset, batch_size, wavelet, wavelet_level, patch_size, num_patches, max_workers, num_data_workers):
    """Preprocess the dataset for a specific wavelet, level, and patch in parallel."""
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_level_wave = {
            executor.submit(preprocess_wavelet, PatchDataset(original_dataset, patch_size, patch_idx),
                             batch_size, wavelet, wavelet_level, num_data_workers): patch_idx
            for patch_idx in range(num_patches)
        }

        for future in as_completed(future_to_level_wave):
            patch_idx = future_to_level_wave[future]
            try:
                results[patch_idx] = future.result()
            except Exception as exc:
                print(f"Patch {patch_idx}, generated an exception: {exc}")

    return results


def fdr_classification(pvalues, threshold=0.05):
    """Perform FDR correction and return whether any p-values are significant."""
    _, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh', alpha=threshold)
    return int(np.any(pvals_corrected < threshold))


def calculate_cdfs_pvalues(real_population_cdfs, input_samples_values):
    """Calculate p-values for input samples against real population CDFs."""
    p_values_per_test = []
    for wavelet_descriptor in sorted(real_population_cdfs.keys()):
        population_cdf_info = real_population_cdfs[wavelet_descriptor]
        sample = input_samples_values[wavelet_descriptor]
        p_value = calculate_p_value_for_sample(sample, population_cdf_info)
        p_values_per_test.append(p_value)
    return p_values_per_test


def main_multiple_patch_test(
    real_population_dataset,
    inference_dataset,
    test_labels=None,
    batch_size=128,
    threshold=0.05,
    patch_size=256,
    wavelet='haar',
    cdf_file="patch_population_cdfs.pkl",
    reload_cdfs=False,
    save_independence_heatmaps=False,
    save_kdes=False,
    ensemble_test='stouffer',
    max_workers=128,
    num_data_workers=2,
    output_dir='logs'
):
    """Run test for number of patches and collect sensitivity and specificity results."""
    print(f"Running test with patch size: {patch_size}")
    wavelet_level = 0

    # Determine number of patches
    example_image, _ = real_population_dataset[0]
    h_patches = example_image.shape[1] // patch_size
    w_patches = example_image.shape[2] // patch_size
    num_patches = h_patches * w_patches

    # Load or compute real population CDFs
    if reload_cdfs and os.path.exists(cdf_file):
        real_population_cdfs = load_population_cdfs(cdf_file)
    else:
        real_population_histograms_per_patch = patch_parallel_preprocess(
            real_population_dataset, batch_size, wavelet, wavelet_level, (patch_size, patch_size), num_patches, max_workers, num_data_workers
        )
        real_population_cdfs = {patch_idx: compute_cdf(histogram) for patch_idx, histogram in real_population_histograms_per_patch.items()}
        # save_population_cdfs(real_population_cdfs, cdf_file)

    # Preprocess inference dataset
    inference_histograms_per_patch = patch_parallel_preprocess(
        inference_dataset, batch_size, wavelet, wavelet_level, (patch_size, patch_size), num_patches, max_workers, num_data_workers
    )
    input_samples = [dict(zip(inference_histograms_per_patch.keys(), values)) for values in zip(*inference_histograms_per_patch.values())]
    input_samples_pvalues = [calculate_cdfs_pvalues(real_population_cdfs, sample) for sample in tqdm(input_samples, desc="Calculating Input's P-values")]

    # Perform ensemble testing
    ensembled_pvalues = perform_ensemble_testing(input_samples_pvalues, ensemble_test)
    predictions = [1 if pval < threshold else 0 for pval in ensembled_pvalues]

    # Save plots
    if save_independence_heatmaps:
        save_independence_test_heatmaps(list(input_samples[0].keys()), predictions, output_dir)

    if save_kdes and test_labels:
        plot_kdes([p for p, l in zip(ensembled_pvalues, test_labels) if l == 0],
                    [p for p, l in zip(ensembled_pvalues, test_labels) if l == 1],
                    os.path.join(output_dir, f"kde_plot_{patch_size}_patches_{ensemble_test}_alpha_{threshold}.png"),
                    "KDE of P-values")

    # Evaluate predictions
    if test_labels:
        metrics = calculate_metrics(test_labels, predictions)
        return metrics


def main_multiple_wavelet_test(
    real_population_dataset,
    inference_dataset,
    test_labels=None,
    batch_size=128,
    threshold=0.05,
    wavelet_list=[],
    cdf_file="wavelet_population_cdfs.pkl",
    reload_cdfs=False,
    save_independence_heatmaps=False,
    save_kdes=False,
    ensemble_test='stouffer',
    max_workers=128,
    num_data_workers=2,
    output_dir='logs',
    max_level=4
):
    """Run test for multiple of wavelets and collect sensitivity and specificity results."""
    print(f"Running test with wavelets: {wavelet_list}")

    # Load or compute real population CDFs
    if reload_cdfs and os.path.exists(cdf_file):
        real_population_cdfs = load_population_cdfs(cdf_file)
    else:
        real_population_histograms_per_wavelet = wave_parallel_preprocess(
            real_population_dataset, batch_size, wavelet_list, list(range(max_level + 1)), max_workers, num_data_workers
        )
        real_population_cdfs = {key: compute_cdf(histogram) for key, histogram in real_population_histograms_per_wavelet.items()}
        # save_population_cdfs(real_population_cdfs, cdf_file)

    # Preprocess inference dataset
    inference_histograms_per_wavelet = wave_parallel_preprocess(
        inference_dataset, batch_size, wavelet_list, list(range(5)), max_workers, num_data_workers
    )
    input_samples = [dict(zip(inference_histograms_per_wavelet.keys(), values)) for values in zip(*inference_histograms_per_wavelet.values())]
    input_samples_pvalues = [calculate_cdfs_pvalues(real_population_cdfs, sample) for sample in tqdm(input_samples, desc="Calculating Input's P-values")]

    # Perform ensemble testing
    ensembled_pvalues = perform_ensemble_testing(input_samples_pvalues, ensemble_test)
    predictions = [1 if pval < threshold else 0 for pval in ensembled_pvalues]

    # Save plots
    if save_independence_heatmaps:
        save_independence_test_heatmaps(list(input_samples[0].keys()), predictions, output_dir)

    if save_kdes and test_labels:
        plot_kdes([p for p, l in zip(ensembled_pvalues, test_labels) if l == 0],
                    [p for p, l in zip(ensembled_pvalues, test_labels) if l == 1],
                    os.path.join(output_dir, f"kde_plot_{len(wavelet_list)*max_level}_wavelets_alpha_{threshold}.png"),
                    "KDE of P-values")

    # Evaluate predictions
    if test_labels:
        metrics = calculate_metrics(test_labels, predictions)
        return metrics


def perform_ensemble_testing(pvalues, ensemble_test):
    """Perform ensemble testing (Stouffer or RBM)."""
    if ensemble_test == 'stouffer':
        return [combine_pvalues(p, method='stouffer')[1] for p in pvalues]
    # elif ensemble_test == 'rbm':
    #     rbm_pipeline = RBMPipeline(n_visible=len(pvalues[0]), n_hidden_list=[1], learning_rate=0.001, epochs=25, batch_size=8)
    #     rbm_pipeline.train(pvalues)
    #     return rbm_pipeline.predict_pvals(pvalues).flatten()
    else:
        raise ValueError(f"Invalid ensemble test: {ensemble_test}")
