from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import warnings
import numpy as np
import torch
from scipy.stats import spearmanr, chi2_contingency, kstest, combine_pvalues, norm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import networkx as nx
from tqdm import tqdm


def calculate_chi2_cremer_v_and_corr(i, j, dist_1, dist_2, bins):
    """Compute Cramér's V and correlation for two distributions."""
    try:
        corr, _ = spearmanr(dist_1, dist_2)
        correlation = abs(corr)
        contingency_table, _, _ = np.histogram2d(dist_1, dist_2, bins=(bins, bins), range=[[0, 1], [0, 1]])
        chi2_stat, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum()
        k = min(contingency_table.shape)
        cramers_v = np.sqrt(chi2_stat / (n * (k - 1)))
        return i, j, cramers_v, correlation
    except ValueError:
        return i, j, -1, correlation


def compute_chi2_and_corr_matrix(keys, distributions, max_workers=128, bins=10):
    """Compute Chi-Square-derived association metrics and correlation matrix."""
    # Stage 2.1: Pairwise dependence testing via chi-square and Cramér's V.
    num_dists = len(distributions)
    chi2_p_matrix = np.zeros((num_dists, num_dists))
    corr_matrix = np.zeros((num_dists, num_dists))

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(num_dists):
            dist_1 = distributions[i]
            for j in range(num_dists):
                if i <= j:
                    continue
                dist_2 = distributions[j]
                tasks.append(executor.submit(calculate_chi2_cremer_v_and_corr, i, j, dist_1, dist_2, bins))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing Chi2 and Correlation tests..."):
            i, j, chi2_p, corr = future.result()
            if chi2_p is not None:
                chi2_p_matrix[i, j] = chi2_p
                chi2_p_matrix[j, i] = chi2_p
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

    return chi2_p_matrix, corr_matrix


def find_largest_independent_group(keys, chi2_p_matrix, p_threshold=0.05, test_type="chi2"):
    """Find a maximal independent group using the Chi-Square p-value matrix."""
    graph = nx.Graph()
    graph.add_nodes_from(keys)

    if test_type == "chi2":
        indices = np.triu(chi2_p_matrix < p_threshold, k=1)
    else:
        masked_p = chi2_p_matrix.copy()
        masked_p[np.tril_indices_from(masked_p)] = 1
        indices = masked_p < p_threshold

    rows, cols = np.where(indices)
    edges = np.column_stack((np.array(keys)[rows], np.array(keys)[cols]))
    graph.add_edges_from(edges)

    subgraph = graph.subgraph([node for node, degree in graph.degree() if degree > 0])
    independent_set = nx.algorithms.approximation.clique.max_clique(subgraph)
    return list(independent_set) if independent_set else [keys[0]]


def find_largest_independent_group_iterative(keys, p_matrix, p_threshold=0.05, test_type="chi2"):
    """Enumerate maximal cliques that satisfy the independence threshold."""
    # Stage 2.2: Independence graph construction from weakly associated statistics.
    graph = nx.Graph()
    graph.add_nodes_from(keys)

    if test_type == "chi2":
        indices = np.triu(p_matrix >= p_threshold, k=1)
    else:
        masked_p = p_matrix.copy()
        masked_p[np.tril_indices_from(masked_p)] = 1
        indices = masked_p < p_threshold

    rows, cols = np.where(indices)
    edges = np.column_stack((np.array(keys)[rows], np.array(keys)[cols]))
    graph.add_edges_from(edges)

    subgraph = graph.subgraph([node for node, degree in graph.degree() if degree > 0])
    return list(nx.find_cliques(subgraph))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remove_nans_from_tests(tests_dict):
    """Filter out tests containing NaN values."""
    cleaned_tests = {}
    for test_name, values in tests_dict.items():
        if np.isnan(values).any():
            warnings.warn(f"Test '{test_name}' contains NaN values and will be excluded.")
        else:
            cleaned_tests[test_name] = values
    return cleaned_tests


def compute_cdf(histogram_values, bins=1000, test_id=None):
    """Compute histogram and cumulative distribution for a statistic."""
    hist, bin_edges = np.histogram(histogram_values, bins=bins, density=True)
    cdf = np.cumsum(hist) * np.diff(bin_edges)
    return hist, bin_edges, cdf


def calculate_metrics(test_labels, predictions):
    """Compute standard classification metrics from predictions."""
    cm = confusion_matrix(test_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
    }


def compute_mean_std_dict(input_dict):
    """Compute mean and standard deviation for each test entry."""
    output_dict = {}

    for key, array in input_dict.items():
        array = np.asarray(array)

        if array.shape[-1] == 1:
            output_dict[key] = array.squeeze()
            continue

        mean_values = np.mean(array, axis=1)
        std_values = np.std(array, axis=1)

        output_dict[f"{key}_mean"] = mean_values
        output_dict[f"{key}_std"] = std_values

    return output_dict


def perform_ensemble_testing(pvalues, ensemble_test):
    """Aggregate p-values according to the requested ensemble method."""
    if ensemble_test == 'stouffer':
        stats = []
        pvals = []
        for p in pvalues:
            stat, pval = combine_pvalues(p, method='stouffer')
            stats.append(stat)
            pvals.append(pval)
        return np.array(stats), np.array(pvals)

    if ensemble_test == 'manual-stouffer':
        pvalues = np.clip(pvalues, np.finfo(np.float32).eps, 1.0 - np.finfo(np.float32).eps)
        inverse_z_scores = norm.ppf(pvalues)
        stouffer_z = np.sum(inverse_z_scores, axis=1) / np.sqrt(pvalues.shape[1])
        stouffer_pvalues = norm.cdf(stouffer_z)
        return stouffer_z, stouffer_pvalues

    if ensemble_test == 'minp':
        pvalues = np.clip(pvalues, np.finfo(np.float32).eps, 1.0 - np.finfo(np.float32).eps)
        min_pvals = np.min(pvalues, axis=1)
        n = pvalues.shape[1]
        aggregated_pvals = 1 - (1 - min_pvals) ** n
        return norm.ppf(aggregated_pvals), aggregated_pvals

    raise ValueError(f"Invalid ensemble test: {ensemble_test}")


def finding_optimal_independent_subgroup_deterministic(
    keys,
    chi2_p_matrix,
    pvals_matrix,
    ensemble_test,
    ks_pvalue_abs_threshold=0.25,
    minimal_p_threshold=0.05,
    preferred_statistics=None,
):
    """Deterministic search over cliques favouring preferred statistics."""
    # Stage 2.3: Maximal clique validation with KS uniformity safeguard.
    preferred_lookup = {
        token.lower()
        for stat in preferred_statistics or ()
        if stat is not None and (token := str(stat).strip())
    }
    preferred_total = len(preferred_lookup)

    candidates = []
    cliques = find_largest_independent_group_iterative(
        keys, chi2_p_matrix, p_threshold=minimal_p_threshold, test_type="cramer_v"
    )

    for clique in tqdm(cliques, total=len(cliques), desc="Searching for optimial clique..."):
        independent_keys_group = list(clique)
        independent_indices = [keys.index(key) for key in independent_keys_group]
        independent_pvals = pvals_matrix[independent_indices].T
        _, ensembled_pvals = perform_ensemble_testing(independent_pvals, ensemble_test)
        ensembled_pvals_subsampled = np.random.choice(ensembled_pvals, size=1000, replace=False)
        _, ks_pvalue = kstest(ensembled_pvals_subsampled, 'uniform', args=(0, 1))
        if abs(ks_pvalue - 0.5) > ks_pvalue_abs_threshold:
            continue

        matched_preferred = {
            preferred_token
            for preferred_token in preferred_lookup
            if any(preferred_token in key.lower() for key in independent_keys_group)
        }
        candidates.append({
            'group': independent_keys_group,
            'size': len(independent_keys_group),
            'ks_pvalue': ks_pvalue,
            'matched_preferred': matched_preferred,
        })

    if not candidates:
        raise ValueError("No valid groups found within the KS p-value range")

    if preferred_total:
        key_fn = lambda c: (len(c['matched_preferred']), c['size'])
    else:
        key_fn = lambda c: (c['size'],)

    best_candidate = max(candidates, key=key_fn)
    best_results = {
        'best_KS': best_candidate['ks_pvalue'],
        'best_N': best_candidate['size'],
        'best_alpha_threshold': minimal_p_threshold,
    }

    if preferred_total:
        preferred_hits = len(best_candidate['matched_preferred'])
        best_results['preferred_hits'] = preferred_hits
        best_results['preferred_total'] = preferred_total
        best_results['preferred_coverage_ratio'] = preferred_hits / preferred_total
        best_results['preferred_missing'] = preferred_total - preferred_hits

    return best_candidate['group'], best_results, None


def balanced_testset(labels, scores, random_state=None):
    """Balance scores across labels by sampling with or without replacement."""
    labels = np.array(labels)
    scores = np.array(scores)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    rng = np.random.default_rng(random_state)

    if len(neg_idx) > len(pos_idx):
        sampled_neg_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)
        sampled_pos_idx = pos_idx
    elif len(pos_idx) > len(neg_idx):
        sampled_neg_idx = rng.choice(neg_idx, size=len(pos_idx), replace=True)
        sampled_pos_idx = pos_idx
    else:
        sampled_neg_idx = neg_idx
        sampled_pos_idx = pos_idx

    balanced_idx = np.concatenate([sampled_pos_idx, sampled_neg_idx])
    balanced_labels = labels[balanced_idx]
    balanced_scores = scores[balanced_idx]

    return balanced_labels, balanced_scores
