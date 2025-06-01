import mlflow
import numpy as np
import pandas as pd
import os
from enum import Enum


class AggregationStrategy(Enum):
    BEST = "best"
    AVERAGE = "average"


def get_auc_summary_per_experiment(
    experiment_name,
    columns_to_export,
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
    auc_column = "metrics.auc" if "metrics.auc" in runs.columns else "metrics.AUC"
    if auc_column not in runs.columns or dataset_key not in runs.columns:
        print(f"Skipping experiment '{experiment_name}' due to missing required columns.")
        return None

    runs["auc"] = np.round(runs[auc_column].fillna(-1), 3)

    grouped = runs.groupby(dataset_key, group_keys=False)

    if aggregation == AggregationStrategy.AVERAGE:
        top_runs = None
        if top_k == 'all':
            top_runs = grouped.apply(lambda x: x.sort_values(by="auc", ascending=False)).reset_index(drop=True)
        else:
            top_runs = grouped.apply(lambda x: x.sort_values(by="auc", ascending=False).head(top_k)).reset_index(drop=True)
        summary = top_runs.groupby(dataset_key)["auc"].agg(['mean', 'std']).round(3).reset_index()

        # Format as "mean ± std"
        summary[experiment_name] = summary.apply(
            lambda row: f"{row['mean']} ± {row['std']}", axis=1
        )
        return summary[[dataset_key, experiment_name]]

    elif aggregation == AggregationStrategy.BEST:
        best_runs = grouped.apply(lambda x: x.loc[x["auc"].idxmax()]).reset_index(drop=True)
        summary = best_runs[[dataset_key, "auc"]].copy()
        summary[experiment_name] = summary["auc"].round(3)
        return summary[[dataset_key, experiment_name]]

    else:
        raise ValueError(f"Unsupported aggregation strategy: {aggregation}")


if __name__ == "__main__":
    # Define experiments and whether to use BEST or AVERAGE aggregation
    experiments = [
        ("R minp patch - Experiments II - Ours", AggregationStrategy.AVERAGE),
        ("R stouffer patch - Experiments II - Ours", AggregationStrategy.AVERAGE),
        ("R minp no_patch - Experiments II - Ours", AggregationStrategy.AVERAGE),
        ("R stouffer no_patch - Experiments II - Ours", AggregationStrategy.AVERAGE),
    ]

    columns_to_export = [
        "params.dataset_type",
        "metrics.AUC"
    ]

    summary_dfs = []
    top_k = 10

    output_file = f'experiments/csvs/final_combined_auc_table{"_all" if top_k == "all" else f"_top_{top_k}" }.csv'
    for experiment_name, aggregation in experiments:
        df = get_auc_summary_per_experiment(
            experiment_name,
            columns_to_export,
            aggregation=aggregation,
            top_k=top_k
        )
        if df is not None:
            summary_dfs.append(df)

    # Merge all summaries on dataset name
    if summary_dfs:
        merged_df = summary_dfs[0]
        for df in summary_dfs[1:]:
            merged_df = pd.merge(merged_df, df, on="params.dataset_type", how="outer")

        merged_df.rename(columns={"params.dataset_type": "Dataset"}, inplace=True)
        merged_df = merged_df.sort_values("Dataset")

        os.makedirs("experiments", exist_ok=True)
        merged_df.to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file}")
    else:
        print("❌ No data exported.")
