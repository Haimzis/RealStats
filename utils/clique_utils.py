"""Utilities for clique finding and related optimizations."""

import optuna
from scipy.stats import kstest
from tqdm import tqdm

from .utils import (
    calculate_chi2,
    calculate_chi2_and_corr,
    compute_chi2_matrix,
    compute_chi2_and_corr_matrix,
    find_largest_independent_group,
    find_largest_independent_group_with_plot,
    find_largest_uncorrelated_group,
    find_largest_independent_group_iterative,
    perform_ensemble_testing,
)


def objective(trial, keys, chi2_p_matrix, pvals_matrix, ensemble_test):
    """Single-objective optimization: minimize |ks_p_value - 0.5|."""
    p_threshold = trial.suggest_float("p_threshold", 0.05, 0.5)
    independent_keys_group = find_largest_independent_group(keys, chi2_p_matrix, p_threshold)
    num_independent_tests = len(independent_keys_group)
    independent_indices = [keys.index(key) for key in independent_keys_group]
    independent_pvals = pvals_matrix[independent_indices].T
    ensembled_stats, _ = perform_ensemble_testing(independent_pvals, ensemble_test)
    _, ks_pvalue = kstest(ensembled_stats, 'norm', args=(0, 1))
    deviation = abs(ks_pvalue - 0.5)
    trial.set_user_attr("independent_keys_group", independent_keys_group)
    trial.set_user_attr("num_independent_tests", num_independent_tests)
    trial.set_user_attr("ks_p_value", ks_pvalue)
    return deviation


def finding_optimal_independent_subgroup(keys, chi2_p_matrix, pvals_matrix, ensemble_test, n_trials=50):
    """Find independent subgroup minimizing KS deviation while maximizing size."""
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, keys, chi2_p_matrix, pvals_matrix, ensemble_test),
                   n_trials=n_trials, show_progress_bar=True)
    valid_trials = [t for t in study.trials if t.value is not None and t.value <= 0.25]
    if not valid_trials:
        raise ValueError("No valid trials found with deviation <= 0.25")
    best_trial = max(valid_trials, key=lambda t: t.user_attrs["num_independent_tests"])
    independent_keys_group = best_trial.user_attrs["independent_keys_group"]
    best_results = {
        'best_KS': best_trial.user_attrs["ks_p_value"],
        'best_N': best_trial.user_attrs["num_independent_tests"],
        'best_alpha_threshold': best_trial.params['p_threshold']
    }
    optimization_data = {
        'thresholds': [t.params["p_threshold"] for t in study.trials if t.value is not None],
        'ks_pvalues': [t.user_attrs["ks_p_value"] for t in study.trials if t.value is not None],
        'num_tests': [t.user_attrs["num_independent_tests"] for t in study.trials if t.value is not None]
    }
    return independent_keys_group, best_results, optimization_data


def uncorrelation_objective(trial, keys, corr_matrix, pvals_matrix, ensemble_test):
    """Objective for uncorrelated subgroup search."""
    p_threshold = trial.suggest_float("p_threshold", 0.0, 0.05)
    independent_keys_group = find_largest_uncorrelated_group(keys, corr_matrix, p_threshold)
    num_independent_tests = len(independent_keys_group)
    independent_indices = [keys.index(key) for key in independent_keys_group]
    independent_pvals = pvals_matrix[independent_indices].T
    ensembled_stats, _ = perform_ensemble_testing(independent_pvals, ensemble_test)
    _, ks_pvalue = kstest(ensembled_stats, 'norm', args=(0, 1))
    deviation = abs(ks_pvalue - 0.5)
    trial.set_user_attr("independent_keys_group", independent_keys_group)
    trial.set_user_attr("num_independent_tests", num_independent_tests)
    trial.set_user_attr("ks_p_value", ks_pvalue)
    return deviation


def finding_optimal_uncorrelated_subgroup(keys, corr_matrix, pvals_matrix, ensemble_test, n_trials=50):
    """Find uncorrelated subgroup via optimization."""
    independent_keys_group = find_largest_uncorrelated_group(keys, corr_matrix, 0.05)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: uncorrelation_objective(t, keys, corr_matrix, pvals_matrix, ensemble_test),
                   n_trials=n_trials, show_progress_bar=True)
    valid_trials = [t for t in study.trials if t.value is not None and t.value <= 0.25]
    if not valid_trials:
        raise ValueError("No valid trials found with deviation <= 0.25")
    best_trial = max(valid_trials, key=lambda t: t.user_attrs["num_independent_tests"])
    independent_keys_group = best_trial.user_attrs["independent_keys_group"]
    best_results = {
        'best_KS': best_trial.user_attrs["ks_p_value"],
        'best_N': best_trial.user_attrs["num_independent_tests"],
        'best_alpha_threshold': best_trial.params['p_threshold']
    }
    optimization_data = {
        'thresholds': [t.params["p_threshold"] for t in study.trials if t.value is not None],
        'ks_pvalues': [t.user_attrs["ks_p_value"] for t in study.trials if t.value is not None],
        'num_tests': [t.user_attrs["num_independent_tests"] for t in study.trials if t.value is not None]
    }
    return independent_keys_group, best_results, optimization_data


def finding_optimal_independent_subgroup_deterministic(
    keys, chi2_p_matrix, pvals_matrix, ensemble_test, fake_pvals_matrix,
    ks_pvalue_abs_threshold=0.25, minimal_p_threshold=0.05):
    """Deterministic search over cliques based on KS range and AUC."""
    best_group = None
    best_results = None
    optimization_data = {'thresholds': [], 'ks_pvalues': [], 'num_tests': []}
    cliques = find_largest_independent_group_iterative(keys, chi2_p_matrix,
                                                       p_threshold=minimal_p_threshold)
    for clique in tqdm(cliques, total=len(cliques), desc="Searching for optimial clique..."):
        independent_keys_group = list(clique)
        num_independent_tests = len(independent_keys_group)
        independent_indices = [keys.index(key) for key in independent_keys_group]
        independent_pvals = pvals_matrix[independent_indices].T
        ensembled_stats, ensembled_pvals = perform_ensemble_testing(independent_pvals, ensemble_test)
        _, ks_pvalue = kstest(ensembled_stats, 'norm', args=(0, 1))
        if abs(ks_pvalue - 0.5) <= ks_pvalue_abs_threshold:
            optimization_data['thresholds'].append(minimal_p_threshold)
            optimization_data['ks_pvalues'].append(ks_pvalue)
            optimization_data['num_tests'].append(num_independent_tests)
            if not best_group or num_independent_tests > best_results['best_N']:
                best_group = independent_keys_group
                best_results = {
                    'best_KS': ks_pvalue,
                    'best_N': num_independent_tests,
                    'best_alpha_threshold': minimal_p_threshold,
                }
    if not best_group:
        raise ValueError("No valid groups found within the KS p-value range")
    return best_group, best_results, optimization_data

__all__ = [
    'calculate_chi2', 'calculate_chi2_and_corr', 'compute_chi2_matrix',
    'compute_chi2_and_corr_matrix', 'find_largest_independent_group',
    'find_largest_independent_group_with_plot', 'find_largest_uncorrelated_group',
    'find_largest_independent_group_iterative',
    'objective', 'finding_optimal_independent_subgroup',
    'uncorrelation_objective', 'finding_optimal_uncorrelated_subgroup',
    'finding_optimal_independent_subgroup_deterministic'
]
