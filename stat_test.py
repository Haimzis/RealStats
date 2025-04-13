import itertools
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from statistics_factory import get_histogram_generator
from utils import (
    AUC_tests_filter,
    compute_cdf,
    calculate_metrics,
    compute_chi2_and_corr_matrix,
    compute_mean_std_dict,
    find_largest_independent_group,
    find_largest_independent_group_iterative,
    find_largest_uncorrelated_group,
    load_population_histograms,
    plot_ks_vs_pthreshold,
    plot_stouffer_analysis,
    plot_uniform_and_nonuniform,
    remove_nans_from_tests,
    save_population_histograms,
    plot_pvalue_histograms,
    plot_binned_histogram,
    plot_histograms,
    save_to_csv,
    split_population_histogram,
    plot_cdf,
    compute_dist_cdf,
    plot_pvalue_histograms_from_arrays
)
from statsmodels.stats.multitest import multipletests
from scipy.stats import combine_pvalues
from data_utils import GlobalPatchDataset, SelfPatchDataset
from scipy.stats import norm, kstest
import optuna
from enum import Enum


class DataType(Enum):
    TRAIN = "train"
    CALIB = "calib"
    TEST = "test"
    TUNING = "tuning"


class TestType(Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    

def get_unique_id(patch_size, level, wave, data_type: DataType):
    """Generate a unique ID for processing."""
    return f"PatchProcessing_wavelet={wave}_level={level}_patch_size={patch_size}_{data_type.value}"


def preprocess_wave(dataset, batch_size, wavelet, wavelet_level, num_data_workers, patch_size, pkl_dir, save_pkl, data_type: DataType):
    """Preprocess the dataset for a single wave level and wavelet type using various histogram statistics."""
    
    # Generate unique filename for saving results
    pkl_filename = os.path.join(pkl_dir, f"{get_unique_id(patch_size, wavelet_level, wavelet, data_type)}.pkl")

    # If not test data and file already exists, load cached histograms
    if data_type != DataType.TEST and os.path.exists(pkl_filename):
        return load_population_histograms(pkl_filename)

    # Create DataLoader for dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_workers)

    # Use the factory function to get the histogram generator
    histogram_generator = get_histogram_generator(wavelet, wavelet_level)

    # If wavelet_level is not valid, return None
    if histogram_generator is None:
        return None

    # Generate histogram
    result = histogram_generator.create_histogram(data_loader)

    # Save results if needed
    if save_pkl:
        save_population_histograms(result, pkl_filename)

    return result


def fdr_classification(pvalues, threshold=0.05):
    """Perform FDR correction and return whether any p-values are significant."""
    _, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh', alpha=threshold)
    return int(np.any(pvals_corrected < threshold))


def calculate_cdfs_pvalues(real_population_cdfs, input_samples_values, test_type=TestType.LEFT):
    """
    Calculate p-values for input samples against real population CDFs.
    """
    p_values_per_test = []
    for descriptor, sample in input_samples_values.items():
        _, bin_edges, population_cdf = real_population_cdfs[descriptor]
        
        # Determine the bin index for each sample
        bin_index = np.clip(np.digitize(sample, bin_edges) - 1, 0, len(population_cdf) - 1)
        pvalue = None 

        if test_type == TestType.LEFT:
            # Only compute left-tailed p-values
            pvalue = population_cdf[bin_index]

        elif test_type == TestType.RIGHT:
            # Only compute right-tailed p-values
            left_p_values = population_cdf[bin_index]
            pvalue = 1 - left_p_values

        elif test_type == TestType.BOTH:
            # Compute two-tailed p-values
            left_p_values = population_cdf[bin_index]
            right_p_values = 1 - left_p_values
            pvalue = np.minimum(left_p_values, right_p_values) * 2

        p_values_per_test.append(pvalue)

    return p_values_per_test


def validate_real_population_cdfs(real_population_cdfs, input_samples):
    """
    Ensure all descriptors in input samples are present in real_population_cdfs.
    """
    sample_descriptors = set(input_samples[0].keys())
    missing_descriptors = sample_descriptors - real_population_cdfs.keys()
    if missing_descriptors:
        raise KeyError(f"Descriptors missing in real_population_cdfs: {missing_descriptors}")


def calculate_pvals_from_cdf(real_population_cdfs, samples_histogram, split="Input's", test_type=TestType.LEFT):
    input_samples = [dict(zip(samples_histogram.keys(), values)) for values in zip(*samples_histogram.values())]
    validate_real_population_cdfs(real_population_cdfs, input_samples)
    input_samples_pvalues = [calculate_cdfs_pvalues(real_population_cdfs, sample, test_type) for sample in tqdm(input_samples, desc=f"Calculating {split} P-values")]
    return input_samples_pvalues


def generate_combinations(patch_sizes, waves, wavelet_levels):
    """Generate all combinations for the training phase."""
    combinations = []
    for patch_size, wavelet, level in itertools.product(patch_sizes, waves, wavelet_levels):
        combinations.append({
            'patch_size': patch_size,
            'wavelet': wavelet,
            'level': level,
        })
    return combinations


def interpret_keys_to_combinations(independent_keys_group):
    """Convert independent keys to relevant test combinations and prevent duplicates."""
    combinations_set = set()
    for key in independent_keys_group:
        match = re.match(r"PatchProcessing_wavelet=([\w.]+)_level=(\d+)_patch_size=(\d+)", key)
        if match:
            wavelet, level, patch_size = match.groups()
            combination = {
                'wavelet': wavelet,
                'level': int(level),
                'patch_size': int(patch_size),
            }
            # Convert the dictionary to a frozenset to use as a hashable type
            combinations_set.add(frozenset(combination.items()))
    
    # Convert the frozensets back to dictionaries
    combinations = [dict(comb) for comb in combinations_set]
    return combinations


def patch_parallel_preprocess(original_dataset, batch_size, combinations, max_workers, num_data_workers, pkl_dir='pkls', save_pkl=False, data_type: DataType = DataType.TRAIN, sort=True):
    """Preprocess the dataset for specific combinations in parallel."""
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_combination = {
            executor.submit(preprocess_wave, SelfPatchDataset(original_dataset, comb['patch_size']),
                            batch_size, comb['wavelet'], comb['level'], num_data_workers, comb['patch_size'], pkl_dir, save_pkl, data_type): comb
            for comb in combinations
        }

        for future in tqdm(as_completed(future_to_combination), total=len(future_to_combination), desc="Processing..."):
            combination = future_to_combination[future]
            unique_id = get_unique_id(combination['patch_size'], combination['level'], combination['wavelet'], data_type)
            try:
                results[unique_id] = future.result()
            except Exception as exc:
                print(f"Combination {combination}, generated an exception: {exc}")

        results = {k: v for k, v in results.items() if v is not None}
        results = {k.replace(f"_{data_type.value}", ""): v for k, v in results.items()}

        if sort:
            results = {k: results[k] for k in sorted(results)}

    return results


def main_multiple_patch_test(
    real_population_dataset,
    fake_population_dataset,
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
    portion=0.2,
    chi2_bins=10,
    cdf_bins=500,
    n_trials=100,
    uniform_p_threshold=0.05,
    calibration_auc_threshold=0.3,
    uniform_sanity_check=False,
    ks_pvalue_abs_threshold=0.25,
    test_type=TestType.LEFT,
    minimal_p_threshold=0.05,
    logger=None,
):
    """Run test for number of patches and collect sensitivity and specificity results."""
    print(f"Running test with: \npatches sizes: {patch_sizes}\nwavelets: {waves}\nlevels: {wavelet_levels}")

    # Generate all combinations for training
    training_combinations = generate_combinations(patch_sizes, waves, wavelet_levels)

    # Load or compute real population histograms
    real_population_histogram = patch_parallel_preprocess(
        real_population_dataset, batch_size, training_combinations, max_workers, num_data_workers, pkl_dir=pkl_dir, save_pkl=True, data_type=DataType.TRAIN
    )

    fake_population_histogram = patch_parallel_preprocess(
        fake_population_dataset, batch_size, training_combinations, max_workers, num_data_workers, pkl_dir=pkl_dir, save_pkl=True, data_type=DataType.CALIB
    )

    real_population_histogram = compute_mean_std_dict(real_population_histogram)
    fake_population_histogram = compute_mean_std_dict(fake_population_histogram)

    real_population_histogram = remove_nans_from_tests(real_population_histogram)
    fake_population_histogram = remove_nans_from_tests(fake_population_histogram)

    real_population_histogram = {k: real_population_histogram[k] for k in real_population_histogram.keys() & fake_population_histogram.keys()}
    fake_population_histogram = {k: fake_population_histogram[k] for k in real_population_histogram.keys() & fake_population_histogram.keys()}

    # Spliting 
    tuning_histogram, training_histogram = split_population_histogram(real_population_histogram, portion)
    tuning_num_samples = len(tuning_histogram[next(iter(tuning_histogram))])
    fake_calibration_histogram, _ = split_population_histogram(fake_population_histogram, tuning_num_samples)
    
    # CDF creation
    real_population_cdfs = {test_id: compute_cdf(values, bins=cdf_bins, test_id=test_id) for test_id, values in training_histogram.items()}    
    
    # Fake Calibration Pvalues
    fake_calibration_pvalues = calculate_pvals_from_cdf(real_population_cdfs, fake_calibration_histogram, DataType.CALIB.name, test_type)
    fake_calibration_pvalues = np.clip(fake_calibration_pvalues, 0, 1)

    # Tuning Pvalues
    tuning_real_population_pvals = calculate_pvals_from_cdf(real_population_cdfs, tuning_histogram, DataType.TUNING.name, test_type)
    tuning_real_population_pvals = np.clip(tuning_real_population_pvals, 0, 1)

    all_keys = list(real_population_histogram.keys())

    auc_scores, best_keys = AUC_tests_filter(tuning_real_population_pvals.T, fake_calibration_pvalues.T, calibration_auc_threshold)
    save_to_csv(np.array(all_keys)[best_keys], auc_scores[best_keys], os.path.join(output_dir, 'individual_auc_scores.csv'))

    fake_calibration_pvalues = fake_calibration_pvalues[:, best_keys]
    tuning_real_population_pvals = tuning_real_population_pvals[:, best_keys]
    tuning_pvalue_distributions = tuning_real_population_pvals.T
    fake_calibration_pvalue_distributions = fake_calibration_pvalues.T
    keys = [all_keys[i] for i in best_keys]

    # for key in keys:
    #     real_histogram = real_population_histogram[key]
    #     fake_histogram = fake_population_histogram[key]
    #     plot_pvalue_histograms(
    #         real_histogram,
    #         fake_histogram,
    #         f'histograms_stats/self_patch/COCO_LEAKAGE_BEST/{key}.png',
    #         title=f"Real and Fake Histogram - {key}",
    #         xlabel='statistic values'
    #     )
    
    if not best_keys.any():
        raise ValueError(f"Fake Calibration Step Error: No individual statistics found with AUC above {calibration_auc_threshold}")

    if uniform_sanity_check:
        # Ignore non uniform distributions
        keys, tuning_real_population_pvals = ks_uniform_sanity_check(
            output_dir, uniform_p_threshold, logger, tuning_real_population_pvals, tuning_pvalue_distributions, keys
        )

    # Chi-Square and Correlation Matrix Computation
    chi2_p_matrix, corr_matrix = compute_chi2_and_corr_matrix(
        keys, tuning_pvalue_distributions, max_workers=max_workers,
        plot_independence_heatmap=save_independence_heatmaps, output_dir=output_dir, bins=chi2_bins
    )
    
    largest_independent_clique_size_approximation = len(find_largest_independent_group(keys, chi2_p_matrix, 0.05))

    independent_keys_group, best_results, optimization_roc = finding_optimal_independent_subgroup_deterministic(
        keys=keys,
        chi2_p_matrix=chi2_p_matrix,
        pvals_matrix=tuning_pvalue_distributions,
        ensemble_test=ensemble_test,
        fake_pvals_matrix=fake_calibration_pvalue_distributions,
        ks_pvalue_abs_threshold=ks_pvalue_abs_threshold,
        minimal_p_threshold=minimal_p_threshold
    )
    
    print(f'Relexation largest clique approximation: {largest_independent_clique_size_approximation}')

    # # Find the Largest Optimal Independent Group using the best p_threshold Fine-tune
    # independent_keys_group, best_results, optimization_roc = finding_optimal_independent_subgroup(
    #     keys=keys,
    #     chi2_p_matrix=chi2_p_matrix,
    #     pvals_matrix=tuning_pvalue_distributions,
    #     ensemble_test=ensemble_test,
    #     n_trials=n_trials,
    # )

    # plot_ks_vs_pthreshold(optimization_roc["thresholds"], optimization_roc["ks_pvalues"], output_dir=output_dir)

    if logger:
        logger.log_param("num_tests", len(real_population_histogram.keys()))
        logger.log_param("Independent keys", independent_keys_group)
        logger.log_metric("largest_independent_clique_size_approximation", largest_independent_clique_size_approximation)
        logger.log_metrics(best_results)

    independent_keys_group_indices = [keys.index(value) for value in independent_keys_group]
    tuning_independent_pvals = tuning_pvalue_distributions[independent_keys_group_indices].T
    perform_ensemble_testing(tuning_independent_pvals, ensemble_test, plot=True, output_dir=output_dir)    
    
    # Convert independent keys to combinations
    independent_combinations = interpret_keys_to_combinations(independent_keys_group)

    # Inference
    inference_histogram = patch_parallel_preprocess(
        inference_dataset, batch_size, independent_combinations, max_workers, num_data_workers, pkl_dir=pkl_dir, save_pkl=False, data_type=DataType.TEST
    )

    inference_histogram = compute_mean_std_dict(inference_histogram)

    inference_histogram = {
        k: inference_histogram[k] for k in independent_keys_group if k in inference_histogram
    }

    input_samples_pvalues = calculate_pvals_from_cdf(real_population_cdfs, inference_histogram, DataType.TEST.name, test_type)
    independent_tests_pvalues = np.array(input_samples_pvalues)
    independent_tests_pvalues = np.clip(independent_tests_pvalues, 0, 1)
        
    ensembled_stats, ensembled_pvalues = perform_ensemble_testing(independent_tests_pvalues, ensemble_test)
    predictions = [1 if pval < threshold else 0 for pval in ensembled_pvalues]

    plot_pvalue_histograms_from_arrays(
        np.array([p for p, l in zip(independent_tests_pvalues, test_labels) if l == 0]),
        np.array([p for p, l in zip(independent_tests_pvalues, test_labels) if l == 1]),
        os.path.join(output_dir, "inference_stat")
        )
    
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


def ks_uniform_sanity_check(output_dir, uniform_p_threshold, logger, tuning_real_population_pvals, pvalue_distributions, keys):
    uniform_tests = []
    ks_pvalues = []

    for i, test_pvalues in tqdm(enumerate(pvalue_distributions), desc="Filtering uniform pvalues distributions", total=len(keys)):
        p_value = kstest(test_pvalues, 'uniform')[1]
        ks_pvalues.append(p_value)
        if p_value > uniform_p_threshold:
            uniform_tests.append(i)

    uniform_keys = [keys[i] for i in uniform_tests]
    uniform_dists = tuning_real_population_pvals[:, uniform_tests]

    # Plots
    plot_uniform_and_nonuniform(pvalue_distributions, uniform_tests, output_dir)
    plot_histograms(ks_pvalues, os.path.join(output_dir, 'ks_pvalues.png'), title='Kolmogorov-Smirnov', bins=20)
     
    if logger:
        logger.log_param("num_uniform_tests", len(uniform_keys))
        logger.log_param("non_uniform_proportion", (len(keys) - len(uniform_keys)) / len(keys))
    return uniform_keys, uniform_dists 


def perform_ensemble_testing(pvalues, ensemble_test, output_dir='logs', plot=False):
    """Perform ensemble testing (Stouffer)."""
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
    elif ensemble_test == 'minp':
        # Ensure p-values are within (0,1) for numerical stability
        pvalues = np.clip(pvalues, np.finfo(np.float32).eps, 1.0 - np.finfo(np.float32).eps)
        min_pvals = np.min(pvalues, axis=1)
        n = pvalues.shape[1]
        # Aggregate p-values using the CDF of the min of n uniform(0,1) variables
        aggregated_pvals = 1 - (1 - min_pvals) ** n
        return norm.ppf(min_pvals), aggregated_pvals
    else:
        raise ValueError(f"Invalid ensemble test: {ensemble_test}")
    

def objective(trial, keys, chi2_p_matrix, pvals_matrix, ensemble_test):
    """
    Single-objective optimization: Minimize abs(ks_p_value - 0.5).
    """
    # Suggest a p_threshold value to test
    p_threshold = trial.suggest_float("p_threshold", 0.05, 0.5)
    
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

    # Calculate the deviation from 0.5 for KS p-value
    deviation = abs(ks_pvalue - 0.5)

    # Store trial attributes for later retrieval
    trial.set_user_attr("independent_keys_group", independent_keys_group)
    trial.set_user_attr("num_independent_tests", num_independent_tests)
    trial.set_user_attr("ks_p_value", ks_pvalue)

    return deviation  # Return deviation to minimize


def finding_optimal_independent_subgroup(keys, chi2_p_matrix, pvals_matrix, ensemble_test, n_trials=50):
    """
    Single-objective optimization to minimize abs(ks_p_value - 0.5) and maximize the number of independent tests.
    """
    # Create a study for single-objective optimization (minimize deviation)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, keys, chi2_p_matrix, pvals_matrix, ensemble_test), n_trials=n_trials, show_progress_bar=True)

    # Filter trials based on deviation (|ks_p_value - 0.5| <= 0.25)
    valid_trials = [
        trial for trial in study.trials
        if trial.value is not None and trial.value <= 0.25
    ]

    if not valid_trials:
        raise ValueError("No valid trials found with deviation <= 0.25")

    # Among valid trials, find the one with the maximum N
    best_trial = max(valid_trials, key=lambda t: t.user_attrs["num_independent_tests"])

    # Extract the best independent group and results
    independent_keys_group = best_trial.user_attrs["independent_keys_group"]
    best_results = {
        'best_KS': best_trial.user_attrs["ks_p_value"],
        'best_N': best_trial.user_attrs["num_independent_tests"],
        'best_alpha_threshold': best_trial.params['p_threshold']
    }

    # Extract optimization data using list comprehensions
    optimization_data = {
        'thresholds': [trial.params["p_threshold"] for trial in study.trials if trial.value is not None],
        'ks_pvalues': [trial.user_attrs["ks_p_value"] for trial in study.trials if trial.value is not None],
        'num_tests': [trial.user_attrs["num_independent_tests"] for trial in study.trials if trial.value is not None]
    }

    print(f"Best State: p_threshold: {best_trial.params['p_threshold']}, KS Pvalue: {best_trial.user_attrs['ks_p_value']}, Num Tests: {best_trial.user_attrs['num_independent_tests']}")
    print(f"Independent Keys Group: {independent_keys_group}")

    return independent_keys_group, best_results, optimization_data


def uncorrelation_objective(trial, keys, corr_matrix, pvals_matrix, ensemble_test):
    """
    Single-objective optimization: Minimize abs(ks_p_value - 0.5).
    """
    # Suggest a p_threshold value to test
    p_threshold = trial.suggest_float("p_threshold", 0.0, 0.05)

    # Step 1: Find the largest independent group
    independent_keys_group = find_largest_uncorrelated_group(keys, corr_matrix, p_threshold)
    num_independent_tests = len(independent_keys_group)
    independent_indices = [keys.index(key) for key in independent_keys_group]

    # Step 2: Extract p-values for independent features
    independent_pvals = pvals_matrix[independent_indices].T

    # Step 3: Perform ensemble testing
    ensembled_stats, _ = perform_ensemble_testing(independent_pvals, ensemble_test)

    # Step 4: Perform KS test against standard normal distribution
    _, ks_pvalue = kstest(ensembled_stats, 'norm', args=(0, 1))  # mean=0, std=1 for standard normal

    # Calculate the deviation from 0.5 for KS p-value
    deviation = abs(ks_pvalue - 0.5)

    # Store trial attributes for later retrieval
    trial.set_user_attr("independent_keys_group", independent_keys_group)
    trial.set_user_attr("num_independent_tests", num_independent_tests)
    trial.set_user_attr("ks_p_value", ks_pvalue)

    return deviation  # Return deviation to minimize


def finding_optimal_uncorrelated_subgroup(keys, corr_matrix, pvals_matrix, ensemble_test, n_trials=50):
    """
    Single-objective optimization to minimize abs(ks_p_value - 0.5) and maximize the number of independent tests.
    """

    independent_keys_group = find_largest_uncorrelated_group(keys, corr_matrix, 0.05)
    print(f'Relexation largest clique approximation: {len(independent_keys_group)}')

    # Create a study for single-objective optimization (minimize deviation)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: uncorrelation_objective(trial, keys, corr_matrix, pvals_matrix, ensemble_test), n_trials=n_trials, show_progress_bar=True)

    # Filter trials based on deviation (|ks_p_value - 0.5| <= 0.25)
    valid_trials = [
        trial for trial in study.trials
        if trial.value is not None and trial.value <= 0.25
    ]

    if not valid_trials:
        raise ValueError("No valid trials found with deviation <= 0.25")

    # Among valid trials, find the one with the maximum N
    best_trial = max(valid_trials, key=lambda t: t.user_attrs["num_independent_tests"])

    # Extract the best independent group and results
    independent_keys_group = best_trial.user_attrs["independent_keys_group"]
    best_results = {
        'best_KS': best_trial.user_attrs["ks_p_value"],
        'best_N': best_trial.user_attrs["num_independent_tests"],
        'best_alpha_threshold': best_trial.params['p_threshold']
    }

    # Extract optimization data using list comprehensions
    optimization_data = {
        'thresholds': [trial.params["p_threshold"] for trial in study.trials if trial.value is not None],
        'ks_pvalues': [trial.user_attrs["ks_p_value"] for trial in study.trials if trial.value is not None],
        'num_tests': [trial.user_attrs["num_independent_tests"] for trial in study.trials if trial.value is not None]
    }

    print(f"Best State: p_threshold: {best_trial.params['p_threshold']}, KS Pvalue: {best_trial.user_attrs['ks_p_value']}, Num Tests: {best_trial.user_attrs['num_independent_tests']}")
    print(f"Independent Keys Group: {independent_keys_group}")

    return independent_keys_group, best_results, optimization_data


def finding_optimal_independent_subgroup_deterministic(keys, chi2_p_matrix, pvals_matrix, ensemble_test, fake_pvals_matrix, ks_pvalue_abs_threshold=0.25, minimal_p_threshold=0.05):
    """
    Deterministic optimization to find the largest independent subgroup
    by iterating over all possible cliques based on KS p-value range and maximum AUC.
    """

    best_group = None
    best_results = None
    optimization_data = {
        'thresholds': [],
        'ks_pvalues': [],
        'num_tests': [],
        'auc_scores': []
    }

    # Find all cliques at the current threshold
    cliques = find_largest_independent_group_iterative(keys, chi2_p_matrix, p_threshold=minimal_p_threshold)
    
    for clique in tqdm(cliques, total=len(cliques), desc="Searching for optimial clique..."):
        # Evaluate each clique
        independent_keys_group = list(clique)
        num_independent_tests = len(independent_keys_group)
        independent_indices = [keys.index(key) for key in independent_keys_group]

        independent_pvals = pvals_matrix[independent_indices].T
        fake_pvals = fake_pvals_matrix[independent_indices].T

        # Perform ensemble testing
        ensembled_stats, ensembled_pvals = perform_ensemble_testing(independent_pvals, ensemble_test)
        fake_ensembled_stats, fake_ensembled_pvals = perform_ensemble_testing(fake_pvals, ensemble_test)

        # Calculate AUC scores
        auc_scores, _ = AUC_tests_filter(ensembled_pvals[np.newaxis, :], fake_ensembled_pvals[np.newaxis, :], auc_threshold=0.0)
        auc = auc_scores.squeeze()

        # Perform KS test
        _, ks_pvalue = kstest(ensembled_stats, 'norm', args=(0, 1))  # mean=0, std=1

        # Filter cliques based on KS p-value range
        if abs(ks_pvalue - 0.5) <= ks_pvalue_abs_threshold:
            optimization_data['thresholds'].append(minimal_p_threshold)
            optimization_data['ks_pvalues'].append(ks_pvalue)
            optimization_data['num_tests'].append(num_independent_tests)
            optimization_data['auc_scores'].append(auc)

            # if not best_group or aux > best_results['best_AUC']:
            if not best_group or num_independent_tests > best_results['best_N']:

                best_group = independent_keys_group
                best_results = {
                    'best_KS': ks_pvalue,
                    'best_N': num_independent_tests,
                    'best_alpha_threshold': minimal_p_threshold,
                    'best_AUC': auc
                }

    if not best_group:
        raise ValueError(f"No valid groups found within the KS p-value range of {0.5 - ks_pvalue_abs_threshold} to {0.5 + ks_pvalue_abs_threshold}")

    print(f"Best State: p_threshold: {best_results['best_alpha_threshold']}, KS Pvalue: {best_results['best_KS']}, Num Tests: {best_results['best_N']}, Max AUC: {best_results['best_AUC']}")
    print(f"Independent Keys Group: {best_group}")

    return best_group, best_results, optimization_data
