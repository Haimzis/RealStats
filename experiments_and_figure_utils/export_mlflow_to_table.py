import mlflow
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
from enum import Enum


class AggregationStrategy(Enum):
    BEST = "best"
    AVERAGE = "average"


def get_summary_per_experiment(
    experiment_name,
    metrics_to_export,
    aggregation=AggregationStrategy.AVERAGE,
    top_k=10
):
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return None

    experiment_id = experiment.experiment_id
    print(f"Processing experiment: {experiment_name} (ID: {experiment_id})")

    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    if runs.empty:
        print(f"No runs found for experiment: {experiment_name}")
        return None

    dataset_key = "params.dataset_type"
    if dataset_key not in runs.columns:
        print(f"Missing column: {dataset_key} in experiment '{experiment_name}'")
        return None

    summary_frames = []
    for metric in metrics_to_export:
        metric_col = f"metrics.{metric}"
        if metric_col not in runs.columns:
            print(f"⚠️ Skipping metric '{metric}' (not found in runs).")
            continue

        temp_runs = runs[[dataset_key, metric_col]].copy()
        temp_runs[metric] = np.round(temp_runs[metric_col].fillna(-1), 3)

        grouped = temp_runs.groupby(dataset_key, group_keys=False)

        if aggregation == AggregationStrategy.AVERAGE:
            # Drop runs where the metric is missing (i.e., failed runs)
            temp_runs = temp_runs.dropna(subset=[metric_col])
            if temp_runs.empty:
                print(f"⚠️ All runs missing metric '{metric}' — skipping.")
                continue

            # Round the metric values
            temp_runs[metric] = np.round(temp_runs[metric_col], 3)

            grouped = temp_runs.groupby(dataset_key, group_keys=False)

            if top_k == "all":
                top_runs = grouped.apply(lambda x: x.sort_values(by=metric, ascending=False)).reset_index(drop=True)
            else:
                # Guard against groups with fewer than top_k valid runs
                top_runs = grouped.apply(lambda x: x.sort_values(by=metric, ascending=False).head(min(top_k, len(x)))).reset_index(drop=True)

            # Compute mean and std
            summary = top_runs.groupby(dataset_key)[metric].agg(['mean', 'std']).round(3).reset_index()

            # Combine into "mean ± std" string
            summary[f"{metric}"] = summary.apply(
                lambda row: f"{row['mean']} ± {row['std']}", axis=1
            )

            summary_frames.append(summary[[dataset_key, f"{metric}"]])


        elif aggregation == AggregationStrategy.BEST:
            best_runs = grouped.apply(lambda x: x.loc[x[metric].idxmax()]).reset_index(drop=True)
            best_runs[f"{experiment_name} - {metric}"] = best_runs[metric].round(3)
            summary_frames.append(best_runs[[dataset_key, f"{experiment_name} - {metric}"]])

        else:
            raise ValueError(f"Unsupported aggregation strategy: {aggregation}")

    if summary_frames:
        merged_summary = summary_frames[0]
        for df in summary_frames[1:]:
            merged_summary = pd.merge(merged_summary, df, on=dataset_key, how="outer")
        return merged_summary

    print(f"No valid metrics found for experiment '{experiment_name}'")
    return None


if __name__ == "__main__":
    # Define experiments and aggregation strategy
    experiments = [
        # ("R minp no patch - Experiments II - With AP", AggregationStrategy.AVERAGE),
        # ("R minp patch dino clip low - Experiments II - With AP", AggregationStrategy.AVERAGE),
        # ("Robustness minp - Jpeg", AggregationStrategy.AVERAGE),
        # ("Robustness minp - Blur", AggregationStrategy.AVERAGE),
        # ("AIStats/minp-no_patch-low-group-leakage-statistics-edit-1", AggregationStrategy.AVERAGE)
        # ("AIStats/minp-no_patch-low_extended", AggregationStrategy.AVERAGE)
        # ("AIStats/minp-no_patch-low_extended-with-prefferation", AggregationStrategy.AVERAGE)
        # ("AIStats/minp-no_patch-low_extended-with-prefferation-only-dino", AggregationStrategy.AVERAGE)
        # ("AIStats/manual-stouffer-no_patch-low_extended-with-prefferation-only-dino", AggregationStrategy.AVERAGE)
        # ("AIStats/minp-no_patch-low-prefer-dino-50per", AggregationStrategy.AVERAGE)
        # ("AIStats/final_all_data_fixed_fake", AggregationStrategy.AVERAGE),
        # ("AIStats/final_all_data_fixed_fake_stf", AggregationStrategy.AVERAGE)
        ("AIStats/final_all_data_all_splits_stf", AggregationStrategy.AVERAGE),
        ("AIStats/final_all_data_all_splits_minp", AggregationStrategy.AVERAGE)
    ]

    metrics_to_export = ["AUC", "AP"]
    top_k = 5

    os.makedirs("experiments/csvs", exist_ok=True)

    for experiment_name, aggregation in experiments:
        output_file = f'experiments/csvs/{experiment_name}_table{"_all" if top_k == "all" else f"_top_{top_k}" }.csv'

        df = get_summary_per_experiment(
            experiment_name,
            metrics_to_export,
            aggregation=aggregation,
            top_k=top_k
        )

        if df is not None:
            df = df.sort_values("params.dataset_type")
            df.to_csv(output_file, index=False)
            print(f"✅ Saved: {output_file}")
        else:
            print(f"❌ No data exported for: {experiment_name}")
