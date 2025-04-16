import mlflow
import numpy as np
import pandas as pd
import os


def export_best_auc_per_dataset(experiment_name, output_csv="experiments_summary.csv", columns_to_export=None):
    """
    Reads a specific experiment from MLflow by name, processes runs, groups by dataset,
    and stores the best AUC per dataset in a CSV file.

    :param experiment_name: Name of the experiment to process.
    :param output_csv: Path to the output CSV file.
    :param columns_to_export: List of columns to include in the CSV file.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    experiment_id = experiment.experiment_id
    print(f"Processing experiment: {experiment_name} (ID: {experiment_id})")

    # Get all runs for the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    if runs.empty:
        print(f"No runs found for experiment: {experiment_name}")
        return

    # Only validate columns that should exist in raw runs
    columns_from_runs = [col for col in columns_to_export if col in runs.columns]
    missing_cols = [col for col in columns_to_export if col not in runs.columns]
    if missing_cols:
        print(f"Skipping experiment '{experiment_name}' due to missing columns: {missing_cols}")
        return

    # Extract column used for grouping (assumed to be 'params.dataset_type')
    if "params.dataset_type" in runs.columns:
        group_key = "params.dataset_type"
    else:
        print(f"Skipping experiment '{experiment_name}' due to missing 'dataset' column for grouping.")
        return

    # Determine AUC metric column
    auc_column = "metrics.auc" if "metrics.auc" in runs.columns else "metrics.AUC"
    if auc_column not in runs.columns:
        print(f"Skipping experiment '{experiment_name}' due to missing AUC metric.")
        return

    # Prepare AUC values
    runs["auc"] = np.round(runs[auc_column].fillna(-1), 5)

    # Group by dataset and select best run per dataset
    grouped = runs.groupby(group_key, group_keys=False)
    best_auc_per_dataset = grouped.apply(lambda x: x.loc[x["auc"].idxmax()]).reset_index(drop=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Select only desired columns
    final_df = best_auc_per_dataset[columns_to_export]

    if "params.waves" in final_df.columns:
        final_df["params.waves"] = final_df["params.waves"].str.replace("RIGID.", "", regex=False)
        
    # Round all numeric columns to 5 decimal places
    for col in final_df.select_dtypes(include=[np.number]).columns:
        final_df[col] = final_df[col].round(5)

    # Save to CSV
    final_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    # List of experiment names
    experiment_names = [
        "RIGID - stouffer test only",
        "RIGID - stouffer test only, limited waves",
        "RIGID statistics - minp",
        "RIGID statistics"
    ]

    # Columns to export
    columns_to_export = [
        "params.dataset_type",
        "metrics.AUC",
        "metrics.best_N",
        "metrics.best_KS",
        "params.patch_sizes",
        "params.waves",
    ]

    for experiment_name in experiment_names:
        safe_name = experiment_name.replace(" ", "_").replace(",", "").replace("-", "").lower()
        output_csv = f"experiments/experiments_summary_{safe_name}.csv"
        export_best_auc_per_dataset(experiment_name, output_csv=output_csv, columns_to_export=columns_to_export)
