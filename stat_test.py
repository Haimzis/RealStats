import itertools
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
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

    if not test and os.path.exists(pkl_filename):
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


def fdr_classification(pvalues, threshold=0.05):
    """Perform FDR correction and return whether any p-values are significant."""
    _, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh', alpha=threshold)
    return int(np.any(pvals_corrected < threshold))


def calculate_cdfs_pvalues(real_population_cdfs, input_samples_values):
    """Calculate p-values for input samples against real population CDFs."""
    p_values_per_test = []
    for descriptor, sample in input_samples_values.items():
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



def generate_combinations(patch_sizes, waves, wavelet_levels, sample_size):
    """Generate all combinations for the training phase."""
    combinations = []
    for patch_size, wavelet, level in itertools.product(patch_sizes, waves, wavelet_levels):
        num_patches = (sample_size[1] // patch_size) * (sample_size[2] // patch_size)
        for patch_index in range(num_patches):
            combinations.append({
                'patch_size': patch_size,
                'wavelet': wavelet,
                'level': level,
                'patch_index': patch_index
            })
    return combinations


def interpret_keys_to_combinations(independent_keys_group):
    """Convert independent keys to relevant test combinations."""
    combinations = []
    for key in independent_keys_group:
        match = re.match(r"PatchProcessing_wavelet=([\w.]+)_level=(\d+)_patch_size=(\d+)_patch_index=(\d+)", key)
        if match:
            wavelet, level, patch_size, patch_index = match.groups()
            combinations.append({
                'wavelet': wavelet,
                'level': int(level),
                'patch_size': int(patch_size),
                'patch_index': int(patch_index)
            })
    return combinations


def patch_parallel_preprocess(original_dataset, batch_size, combinations, max_workers, num_data_workers, pkl_dir='pkls', save_pkl=False, test=False, sort=True):
    """Preprocess the dataset for specific combinations in parallel."""
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_combination = {
            executor.submit(preprocess_wave, PatchDataset(original_dataset, comb['patch_size'], comb['patch_index']),
                            batch_size, comb['wavelet'], comb['level'], num_data_workers, comb['patch_size'], comb['patch_index'], pkl_dir, save_pkl, test): comb
            for comb in combinations
        }

        for future in tqdm(as_completed(future_to_combination), total=len(future_to_combination), desc="Processing..."):
            combination = future_to_combination[future]
            unique_id = get_unique_id(combination['patch_size'], combination['patch_index'], combination['level'], combination['wavelet'], test)
            try:
                results[unique_id] = future.result()
            except Exception as exc:
                print(f"Combination {combination}, generated an exception: {exc}")

        results = {k: v for k, v in results.items() if v is not None}

        if test:
            results = {k.replace('_test', ''): v for k, v in results.items() if v is not None}

        if sort:
            results = {k: results[k] for k in sorted(results)}
    return results


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
    portion=0.05,
    criteria='KS',
    chi2_bins=201,
    n_trials=100,
    logger=None
):
    """Run test for number of patches and collect sensitivity and specificity results."""
    print(f"Running test with: \npatches sizes: {patch_sizes}\nwavelets: {waves}\nlevels: {wavelet_levels}")

    # Determine number of patches
    example_image, _ = real_population_dataset[0]

    # Generate all combinations for training
    training_combinations = generate_combinations(patch_sizes, waves, wavelet_levels, example_image.shape)

    # Load or compute real population histograms
    real_population_histogram = patch_parallel_preprocess(
        real_population_dataset, batch_size, training_combinations, max_workers, num_data_workers, pkl_dir=pkl_dir, save_pkl=True, test=False
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
        plot_independence_heatmap=save_independence_heatmaps, output_dir=output_dir, bins=chi2_bins
    )

    # Find the Largest Optimal Independent Group using the best p_threshold Fine-tune
    independent_keys_group, results = finding_optimal_independent_subgroup(
        keys=keys,
        chi2_p_matrix=chi2_p_matrix,
        pvals_matrix=distributions,
        ensemble_test=ensemble_test,
        n_trials=n_trials,
        criteria=criteria
    )

    if logger:
        logger.log_param("num_tests", len(real_population_histogram.keys()))
        logger.log_param("Independent keys", independent_keys_group)
        logger.log_metrics(results)

    independent_keys_group_indices = [keys.index(value) for value in independent_keys_group]
    tuning_independent_pvals = distributions[independent_keys_group_indices].T
    perform_ensemble_testing(tuning_independent_pvals, ensemble_test, plot=True, output_dir=output_dir)    
    
    # Convert independent keys to combinations
    independent_combinations = interpret_keys_to_combinations(independent_keys_group)

    # Inference
    inference_histogram = patch_parallel_preprocess(
        inference_dataset, batch_size, independent_combinations, max_workers, num_data_workers, pkl_dir=pkl_dir, save_pkl=False, test=True
    )

    input_samples_pvalues = calculate_pvals_from_cdf(real_population_cdfs, inference_histogram, "Test")
    independent_tests_pvalues = np.array(input_samples_pvalues)
    independent_tests_pvalues = np.clip(independent_tests_pvalues, 0, 1)

    # Perform ensemble testing
    ensembled_stats, ensembled_pvalues = perform_ensemble_testing(independent_tests_pvalues, ensemble_test)
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


def perform_ensemble_testing(pvalues, ensemble_test, output_dir='logs', plot=False):
    """Perform ensemble testing (Stouffer or RBM)."""
    if ensemble_test == 'stouffer':
        return [combine_pvalues(p, method='stouffer')[1] for p in pvalues]
    elif ensemble_test == 'manual-stouffer':
        pvalues = np.clip(pvalues, np.finfo(np.float32).eps, 1.0 - np.finfo(np.float32).eps)
        inverse_z_scores = norm.ppf(pvalues)
        stouffer_z = np.sum(inverse_z_scores, axis=1) / np.sqrt(pvalues.shape[1])
        stouffer_pvalues = norm.cdf(stouffer_z)
        if plot:
            plot_stouffer_analysis(pvalues, inverse_z_scores, stouffer_z, stouffer_pvalues, num_plots_pvalues=5, num_plots_zscores=5, output_folder=output_dir)
        return stouffer_z, stouffer_pvalues
    else:
        raise ValueError(f"Invalid ensemble test: {ensemble_test}")
    

def objective(trial, keys, chi2_p_matrix, pvals_matrix, ensemble_test):
    """
    Multi-objective optimization: maximize KS pvalue and maximize independent test count.
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
    _, ks_pvalue = kstest(ensembled_stats, 'norm', args=(0, 1))  # mean=0, std=1 for standard normal

    trial.set_user_attr("independent_keys_group", independent_keys_group)

    return ks_pvalue, num_independent_tests


def finding_optimal_independent_subgroup(keys, chi2_p_matrix, pvals_matrix, ensemble_test, n_trials=50, criteria='KS'):
    """
    Multi-objective optimization of p_threshold using Optuna to maximize KS pvalue
    and maximize the number of independent tests.
    """
    # criteria selection
    sorting_order = lambda trial: (trial.values[1], trial.values[0]) if criteria != 'KS' else (trial.values[0], trial.values[1]) 

    # Create a study for multi-objective optimization
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(lambda trial: objective(trial, keys, chi2_p_matrix, pvals_matrix, ensemble_test),
                   n_trials=n_trials, show_progress_bar=True)

    # Return the best trial based on your preference
    best_trial = sorted(study.best_trials, key=lambda t: sorting_order(t), reverse=True)[0] # prioritize KS statistic minimization
    independent_keys_group = best_trial.user_attrs["independent_keys_group"]
    print(f"Best State: p_threshold: {best_trial.params['p_threshold']}, KS Pvalue: {best_trial.values[0]}, Num Tests: {best_trial.values[1]}")
    print(f"Independent Keys Group: {independent_keys_group}")

    results = {'best_KS': best_trial.values[0], 'best_N': best_trial.values[1], 'best_alpha_threshold': best_trial.params['p_threshold']}
    return independent_keys_group, results