import mlflow
import numpy as np
import pandas as pd
import os
from enum import Enum


class AggregationStrategy(Enum):
    BEST = "best"
    AVERAGE = "average"


def export_auc_per_dataset(
    experiment_name,
    output_csv="experiments_summary.csv",
    columns_to_export=None,
    aggregation=AggregationStrategy.BEST,
):
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    experiment_id = experiment.experiment_id
    print(f"Processing experiment: {experiment_name} (ID: {experiment_id})")

    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    if runs.empty:
        print(f"No runs found for experiment: {experiment_name}")
        return

    columns_from_runs = [col for col in columns_to_export if col in runs.columns]
    missing_cols = [col for col in columns_to_export if col not in runs.columns]
    if missing_cols:
        print(f"Skipping experiment '{experiment_name}' due to missing columns: {missing_cols}")
        return

    if "params.dataset_type" in runs.columns:
        group_key = "params.dataset_type"
    else:
        print(f"Skipping experiment '{experiment_name}' due to missing 'dataset' column for grouping.")
        return

    auc_column = "metrics.auc" if "metrics.auc" in runs.columns else "metrics.AUC"
    if auc_column not in runs.columns:
        print(f"Skipping experiment '{experiment_name}' due to missing AUC metric.")
        return

    runs["auc"] = np.round(runs[auc_column].fillna(-1), 3)

    grouped = runs.groupby(group_key, group_keys=False)

    if aggregation == AggregationStrategy.BEST:
        processed = grouped.apply(lambda x: x.loc[x["auc"].idxmax()]).reset_index(drop=True)
        final_df = processed[columns_to_export]

    elif aggregation == AggregationStrategy.AVERAGE:
        numeric_cols = runs.select_dtypes(include=[np.number]).columns
        agg = grouped[numeric_cols].agg(['mean', 'std'])

        # Flatten MultiIndex columns
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        agg = agg.reset_index()

        # Bring in first categorical columns (excluding the group key)
        non_numeric_cols = [col for col in columns_to_export if col not in numeric_cols and col != group_key]
        for col in non_numeric_cols:
            if col in runs.columns:
                first_values = grouped[col].first().reset_index()
                agg = pd.merge(agg, first_values, on=group_key, how="left")

        final_df = agg

    else:
        raise ValueError(f"Unsupported aggregation strategy: {aggregation}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Define which output columns to keep
    output_columns = []
    for col in columns_to_export:
        if aggregation == AggregationStrategy.BEST:
            output_columns.append(col)
        else:
            if col in runs.select_dtypes(include=[np.number]).columns:
                output_columns.append(f"{col}_mean")
                output_columns.append(f"{col}_std")
            else:
                output_columns.append(col)

    final_df = final_df[output_columns]

    if "params.waves" in final_df.columns:
        final_df["params.waves"] = final_df["params.waves"].str.replace("RIGID.", "", regex=False)

    # Round numeric columns
    for col in final_df.select_dtypes(include=[np.number]).columns:
        final_df[col] = final_df[col].round(3)

    final_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    experiment_names = [
        "R minp no patch - Experiments II",
    ]

    columns_to_export = [
        "params.dataset_type",
        "metrics.AUC",
        # "metrics.best_N",
        # "metrics.best_KS",
        # "params.patch_sizes",
        # "params.waves",
    ]

    for experiment_name in experiment_names:
        safe_name = experiment_name.replace(" ", "_").replace(",", "").replace("-", "").lower()

        # output_csv_best = f"experiments/experiments_summary_{safe_name}_best.csv"
        # export_auc_per_dataset(
        #     experiment_name,
        #     output_csv=output_csv_best,
        #     columns_to_export=columns_to_export,
        #     aggregation=AggregationStrategy.BEST,
        # )

        output_csv_avg = f"experiments/experiments_summary_{safe_name}_average.csv"
        export_auc_per_dataset(
            experiment_name,
            output_csv=output_csv_avg,
            columns_to_export=columns_to_export,
            aggregation=AggregationStrategy.AVERAGE,
        )
