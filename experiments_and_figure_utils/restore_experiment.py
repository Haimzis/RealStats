import mlflow
import json
import os
import ast

EXPERIMENT_ID = "869866565155270511"
OUTPUT_JSON_PATH = "configs/AIStats/robustness/final_jpeg.json"

FIXED_ARGS = [
    "--patch_divisors",
    "--dataset_type",
    "--seed",
    "--sample_size",
    "--threshold",
    "--save_histograms",
    "--save_independence_heatmaps",
    "--pkls_dir",
    "--run_id",
]

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

# Load runs from MLflow
runs = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID],
                          filter_string='attributes.status != "FAILED"')
if runs.empty:
    print(f"No runs found for experiment ID: {EXPERIMENT_ID}")
    exit()

config_lines = []

for i, (_, run) in enumerate(runs.iterrows()):
    cli_args = []

    # Handle independent keys as space-separated string
    if "params.Independent keys" in run:
        try:
            keys_list = ast.literal_eval(run["params.Independent keys"])
            if isinstance(keys_list, list):
                keys_string = " ".join(keys_list)
                cli_args.append(f"--independent_keys {keys_string}")
            else:
                raise ValueError("Not a list")
        except Exception as e:
            print(f"Skipping run {i}: Invalid 'params.Independent keys' format - {e}")
            continue
    else:
        print(f"Skipping run {i}: Missing 'params.Independent keys'")
        continue

    seed = run["params.seed"]
    # Append all other CLI arguments
    for arg in FIXED_ARGS:
        key = f"params.{arg.lstrip('--')}"
        if key in run:
            value = run[key]
            # Remove brackets if it's a stringified list
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    value = " ".join(map(str, parsed))
            except:
                pass
            cli_args.append(f"{arg} {value}")

    config_line = " ".join(cli_args).replace("seed=42", f"seed={seed}")
    config_lines.append(config_line)

# Write to JSON file
with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(config_lines, f, indent=2)

print(f"✅ Saved {len(config_lines)} configurations to {OUTPUT_JSON_PATH}")
