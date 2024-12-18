import itertools
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from processing.histograms import DCTNormHistogram, FourierNormHistogram, WaveletNormHistogram
from utils import (
    compute_cdf,
    calculate_metrics,
    compute_chi2_and_corr_matrix,
    find_largest_independent_group,
    load_population_histograms,
    plot_stouffer_analysis,
    save_population_histograms,
    plot_pvalue_histograms,
    plot_histograms
)
from statsmodels.stats.multitest import multipletests
from scipy.stats import combine_pvalues
from data_utils import PatchDataset
from scipy.stats import norm, kstest
import optuna


# TODO: 
# 2. make many many more tests [all levels, all waves, all patches, preprocessing]
# 4. baselines table [AUC, 5 methods, 5 generators]
def get_unique_id(patch_size, patch_idx, level, wave, test):
    return f"PatchProcessing_wavelet={wave}_level={level}_patch_size={patch_size}_patch_index={patch_idx}{'_test' if test else ''}"


def preprocess_wave(dataset, batch_size, wavelet, wavelet_level, num_data_workers, patch_size, patch_index, pkl_dir, save_pkl, test=False):
    """Preprocess the dataset for a single wave level and wavelet type using NormHistogram."""
    pkl_filename = os.path.join(pkl_dir, f"{get_unique_id(patch_size, patch_index, wavelet_level, wavelet, test)}.pkl")

    if os.path.exists(pkl_filename):
        return load_population_histograms(pkl_filename)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_workers)
    if wavelet in ['bior1.1', 'bior3.1', 'bior6.8', 'coif1', 'coif10', 'db1', 'db38', 'haar', 'rbio6.8', 'sym2']:
        histogram_generator = WaveletNormHistogram(selected_indices=[wavelet_level], wave=wavelet)
    elif wavelet == 'fourier':
        if wavelet_level != 0:
            return None
        histogram_generator = FourierNormHistogram()
    elif wavelet == 'dct':
        if wavelet_level > 4 or wavelet_level == 0: 
            return None
        histogram_generator = DCTNormHistogram(dct_type=wavelet_level)
    else:
        raise ValueError('Invalid wave type.')

    result = histogram_generator.create_histogram(data_loader)
    if save_pkl:
        save_population_histograms(result, pkl_filename)
    return result


def patch_parallel_preprocess(original_dataset, batch_size, waves, wavelet_levels, patch_sizes, sample_size, max_workers, num_data_workers, pkl_dir='pkls', save_pkl=False, test=False, sort=True):
    """Preprocess the dataset for a specific wavelet, level, and patch in parallel."""
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_level_wave = {
            executor.submit(preprocess_wave, PatchDataset(original_dataset, patch_size, patch_idx),
                            batch_size, wavelet, level, num_data_workers, patch_size, patch_idx, pkl_dir, save_pkl, test): (patch_size, patch_idx, level, wavelet)
            for patch_size in patch_sizes
            for wavelet, level in itertools.product(waves, wavelet_levels)
            for patch_idx in range((sample_size[1] // patch_size) * (sample_size[2] // patch_size))
        }        

        for future in tqdm(as_completed(future_to_level_wave), total=len(future_to_level_wave), desc="Processing..."):
            patch_size, patch_idx, level, wave = future_to_level_wave[future]
            unique_id = get_unique_id(patch_size, patch_idx, level, wave, test)
            try:
                results[unique_id] = future.result()
            except Exception as exc:
                print(f"Patch {patch_idx}, generated an exception: {exc}")

        results = {k: v for k, v in results.items() if v is not None}

        if test:
            results = {k.replace('_test', ''): v for k, v in results.items() if v is not None}

        if sort is True: 
            results = {k: results[k] for k in sorted(results)}
    return results


def fdr_classification(pvalues, threshold=0.05):
    """Perform FDR correction and return whether any p-values are significant."""
    _, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh', alpha=threshold)
    return int(np.any(pvals_corrected < threshold))


def calculate_cdfs_pvalues(real_population_cdfs, input_samples_values):
    """Calculate p-values for input samples against real population CDFs."""
    p_values_per_test = []
    for descriptor in real_population_cdfs.keys():
        sample = input_samples_values[descriptor]
        _, bin_edges, population_cdf = real_population_cdfs[descriptor]
        bin_index = np.clip(np.digitize(sample, bin_edges) - 1, 0, len(population_cdf) - 1)
        p_values_per_test.append(population_cdf[bin_index])        
    return p_values_per_test


def validate_real_population_cdfs(real_population_cdfs, input_samples):
    """
    Ensure all descriptors in input samples are present in real_population_cdfs.
    """
    sample_descriptors = set(input_samples[0].keys())
    missing_descriptors = sample_descriptors - real_population_cdfs.keys()
    if missing_descriptors:
        raise KeyError(f"Descriptors missing in real_population_cdfs: {missing_descriptors}")


def calculate_pvals_from_cdf(real_population_cdfs, samples_histogram, split="Input's"):
    input_samples = [dict(zip(samples_histogram.keys(), values)) for values in zip(*samples_histogram.values())]
    validate_real_population_cdfs(real_population_cdfs, input_samples)
    input_samples_pvalues = [calculate_cdfs_pvalues(real_population_cdfs, sample) for sample in tqdm(input_samples, desc=f"Calculating {split} P-values")]
    return input_samples_pvalues


def main_multiple_patch_test(
    real_population_dataset,
    inference_dataset,
    waves,
    wavelet_levels,
    patch_sizes,
    test_labels=None,
    batch_size=128,
    threshold=0.05,
    save_independence_heatmaps=False,
    save_histograms=False,
    ensemble_test='stouffer',
    max_workers=128,
    num_data_workers=2,
    output_dir='logs',
    pkl_dir='pkls',
    return_logits=False,
    portion=0.05
):
    """Run test for number of patches and collect sensitivity and specificity results."""
    print(f"Running test with: \npatches sizes: {patch_sizes}\nwavelets: {waves}\nlevels: {wavelet_levels}")

    # Determine number of patches
    example_image, _ = real_population_dataset[0]

    # Load or compute real population histograms
    real_population_histogram = patch_parallel_preprocess(
        real_population_dataset, batch_size, waves, wavelet_levels, patch_sizes, example_image.shape, max_workers, num_data_workers, pkl_dir=pkl_dir, save_pkl=True
    )

    # Spliting 
    population_length = len(list(real_population_histogram.values())[0])
    split_point = int(population_length * portion) # Split the population into tuning and training portions
    tuning_histogram = {k: v[:split_point] for k, v in real_population_histogram.items()}
    training_histogram = {k: v[split_point:] for k, v in real_population_histogram.items()}

    real_population_cdfs = {test_id: compute_cdf(values) for test_id, values in training_histogram.items()}
    tuning_real_population_pvals = calculate_pvals_from_cdf(real_population_cdfs, tuning_histogram, "Tuning")
    tuning_real_population_pvals = np.clip(tuning_real_population_pvals, 0, 1)

    # Chi-Square and Correlation Matrix Computation
    keys = list(real_population_histogram.keys())
    distributions = np.array(tuning_real_population_pvals).T

    chi2_p_matrix, corr_matrix = compute_chi2_and_corr_matrix(
        keys, distributions, max_workers=max_workers,
        plot_independence_heatmap=save_independence_heatmaps, output_dir=output_dir
    )

    # Fine-tune p_threshold using KS test
    best_p_threshold = fine_tune_p_threshold_with_optuna(
        keys=keys,
        chi2_p_matrix=chi2_p_matrix,
        pvals_matrix=distributions,
        ensemble_test=ensemble_test,
        n_trials=50
    )

    # Find the Largest Independent Group using the best p_threshold
    independent_keys_group = find_largest_independent_group(keys, chi2_p_matrix, p_threshold=best_p_threshold)
    independent_keys_group_indices = [keys.index(value) for value in independent_keys_group]

    print(f'Found {len(independent_keys_group_indices)} independent tests.')

    # Inference
    inference_histogram = patch_parallel_preprocess(
        inference_dataset, batch_size, waves, wavelet_levels, patch_sizes, example_image.shape, max_workers, num_data_workers, pkl_dir=pkl_dir, save_pkl=True, test=True
    )

    input_samples_pvalues = calculate_pvals_from_cdf(real_population_cdfs, inference_histogram, "Test")
    independent_tests_pvalues = np.array(input_samples_pvalues).T[independent_keys_group_indices].T
    independent_tests_pvalues = np.clip(independent_tests_pvalues, 0, 1)

    # Perform ensemble testing
    ensembled_stats, ensembled_pvalues = perform_ensemble_testing(independent_tests_pvalues, ensemble_test, plot=True)
    predictions = [1 if pval < threshold else 0 for pval in ensembled_pvalues]

    if save_histograms and test_labels:
        plot_pvalue_histograms(
            [p for p, l in zip(ensembled_pvalues, test_labels) if l == 0],
            [p for p, l in zip(ensembled_pvalues, test_labels) if l == 1],
            os.path.join(output_dir, f"histogram_plot_{patch_sizes}_{ensemble_test}_alpha_{threshold}.png"),
            "Histogram of P-values"
        )
    
    if return_logits:
        return {
            'scores': 1 - np.array(ensembled_pvalues),
            'n_tests': len(list(independent_keys_group))
        }

    # Evaluate predictions
    if test_labels:
        metrics = calculate_metrics(test_labels, predictions)
        return metrics


def perform_ensemble_testing(pvalues, ensemble_test, plot=False):
    """Perform ensemble testing (Stouffer or RBM)."""
    if ensemble_test == 'stouffer':
        return [combine_pvalues(p, method='stouffer')[1] for p in pvalues]
    elif ensemble_test == 'manual-stouffer':
        pvalues = np.clip(pvalues, np.finfo(np.float32).eps, 1.0 - np.finfo(np.float32).eps)
        inverse_z_scores = norm.ppf(pvalues)
        stouffer_z = np.sum(inverse_z_scores, axis=1) / np.sqrt(pvalues.shape[1])
        stouffer_pvalues = norm.cdf(stouffer_z)
        if plot:
            plot_stouffer_analysis(pvalues, inverse_z_scores, stouffer_z, stouffer_pvalues, num_plots_pvalues=5, num_plots_zscores=5)
        return stouffer_z, stouffer_pvalues
    else:
        raise ValueError(f"Invalid ensemble test: {ensemble_test}")
    

def objective(trial, keys, chi2_p_matrix, pvals_matrix, ensemble_test):
    """
    Multi-objective optimization: minimize KS statistic and maximize independent test count.
    """
    # Suggest a p_threshold value to test
    p_threshold = trial.suggest_float("p_threshold", 0.05, 0.5, log=True)
    
    # Step 1: Find the largest independent group
    independent_keys_group = find_largest_independent_group(keys, chi2_p_matrix, p_threshold)
    num_independent_tests = len(independent_keys_group)
    independent_indices = [keys.index(key) for key in independent_keys_group]

    # Step 2: Extract p-values for independent features
    independent_pvals = pvals_matrix[independent_indices].T

    # Step 3: Perform ensemble testing
    ensembled_stats, _ = perform_ensemble_testing(independent_pvals, ensemble_test)

    # Step 4: Perform KS test against standard normal distribution
    ks_stat, _ = kstest(ensembled_stats, 'norm', args=(0, 1))  # mean=0, std=1 for standard normal

    # Return two objectives: KS statistic and negative number of independent tests (maximize by negating)
    return ks_stat, num_independent_tests


def fine_tune_p_threshold_with_optuna(keys, chi2_p_matrix, pvals_matrix, ensemble_test, n_trials=50):
    """
    Multi-objective optimization of p_threshold using Optuna to minimize KS statistic
    and maximize the number of independent tests.
    """
    # Create a study for multi-objective optimization
    study = optuna.create_study(directions=["minimize", "maximize"])
    study.optimize(lambda trial: objective(trial, keys, chi2_p_matrix, pvals_matrix, ensemble_test),
                   n_trials=n_trials, show_progress_bar=True)

    # Return the best trial based on your preference
    best_trial = sorted(study.best_trials, key=lambda t: (-t.values[1], t.values[0]))[0] # prioritize KS statistic minimization
    print(f"Best State: p_threshold: {best_trial.params['p_threshold']}, KS Statistic: {best_trial.values[0]}, Num Tests: {best_trial.values[1]}")
    return best_trial.params['p_threshold']