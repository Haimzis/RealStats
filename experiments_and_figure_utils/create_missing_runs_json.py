import mlflow
import pandas as pd
import os
import json
import argparse

def find_crashed_seeds(experiment_id: str, auc_keys=("metrics.auc", "metrics.AUC")) -> set:
    print(f"🔍 Searching runs in experiment ID: {experiment_id}")
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    if runs.empty:
        print("⚠️ No runs found.")
        return set()

    auc_col = next((k for k in auc_keys if k in runs.columns), None)
    if not auc_col:
        print("⚠️ No AUC column found.")
        return set()

    # Crashed = AUC is NaN
    crashed_runs = runs[runs[auc_col].isna()]
    if "params.seed" not in crashed_runs.columns:
        print("⚠️ Missing 'params.seed' in run parameters.")
        return set()

    seeds = set(crashed_runs["params.seed"].dropna().astype(int))
    print(f"💥 Found {len(seeds)} crashed seeds.")
    return seeds


def extract_missing_configurations(json_path: str, crashed_seeds: set, output_path: str):
    if not os.path.isfile(json_path):
        print(f"❌ Config JSON not found: {json_path}")
        return

    with open(json_path, "r") as f:
        config_lines = json.load(f)

    print(f"🧾 Loaded {len(config_lines)} config lines from: {json_path}")

    matched_lines = [
        line for line in config_lines
        if any(f"--seed {seed}" in line for seed in crashed_seeds)
    ]

    print(f"✅ Matched {len(matched_lines)} missing config lines.")

    with open(output_path, "w") as f:
        json.dump(matched_lines, f, indent=2)

    print(f"📁 Saved missing configurations to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find crashed MLflow runs and recover config lines.")
    parser.add_argument("--experiment_id", help="MLflow experiment ID")
    parser.add_argument("--config_json_path", help="Path to the experiment configuration JSON file")
    args = parser.parse_args()

    experiment_id = args.experiment_id
    json_path = args.config_json_path
    output_path = os.path.splitext(json_path)[0] + "_missing.json"

    crashed_seeds = find_crashed_seeds(experiment_id)
    if crashed_seeds:
        extract_missing_configurations(json_path, crashed_seeds, output_path)
    else:
        print("🎉 No crashed seeds found. You're all good!")
