import os
import random
import json
import argparse

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from datasets_factory import DatasetType
from statistics_factory import STATISTIC_HISTOGRAMS
from utils import build_backbones_statistics_list

# Define parameter ranges
finetune_portion_range = [0.1]  # treated as list for consistency
# dataset_types = [
#     'BIGGAN_TEST_ONLY', 'CYCLEGAN_TEST_ONLY', 'GAUGAN_TEST_ONLY', 'PROGAN_TEST_ONLY',
#     'SEEINGDARK_TEST_ONLY', 'STYLEGAN_TEST_ONLY', 'CRN_TEST_ONLY', 'DEEPFAKE_TEST_ONLY',
#     'IMLE_TEST_ONLY', 'SAN_TEST_ONLY', 'STARGAN_TEST_ONLY', 'STYLEGAN2_TEST_ONLY',
#     'CELEBA_TEST_ONLY', 'COCO_TEST_ONLY', 'COCO_BIGGAN_256_TEST_ONLY',
#     'COCO_STABLE_DIFFUSION_XL_TEST_ONLY', 'COCO_DALLE3_COCOVAL_TEST_ONLY',
#     'COCO_SYNTH_MIDJOURNEY_V5_TEST_ONLY', 'COCO_STABLE_DIFFUSION_2_TEST_ONLY'
# ]

dataset_types = [
    member.name for member in DatasetType
    if "GROUP_LEAKAGE" in member.name
]

statistics_choices = [k for k in STATISTIC_HISTOGRAMS if k.startswith("RIGID.") and any(k.endswith(suffix) for suffix in [".05", ".10"])]

patch_divisors_choices = ["0"]
chi2_bins_choices = [15]
statistic_ensemble = "minp" 
kspvalue_abs_thresholds = [0.45]  
minimal_p_thresholds = [0.07]

RUNS_PER_CONFIG = 15

def generate_configurations(runs_per_config):
    configs = []
    used_dataset_types = set()

    # Force each dataset_type to appear at least once
    for dataset_type in dataset_types:
        finetune_portion = random.choice(finetune_portion_range)
        statistics = statistics_choices
        patch_divisors = random.choice(patch_divisors_choices)
        chi2_bins = random.choice(chi2_bins_choices)
        kspvalue_abs_threshold = random.choice(kspvalue_abs_thresholds)
        minimal_p_threshold = random.choice(minimal_p_thresholds)

        base_config = (
            f"--finetune_portion {finetune_portion} "
            f"--statistics {' '.join(statistics)} "
            f"--ensemble_test {statistic_ensemble} "
            f"--patch_divisors {patch_divisors} "
            f"--chi2_bins {chi2_bins} "
            f"--dataset_type {dataset_type} "
            f"--pkls_dir pkls/AIStats/new_stats "
            f"--cdf_bins 400 "
            f"--ks_pvalue_abs_threshold {kspvalue_abs_threshold} "
            f"--minimal_p_threshold {minimal_p_threshold} "
            f"--experiment_id AIStats/minp-no_patch-low" 
        )

        for i in range(runs_per_config):
            full_config = f"{base_config} --seed {i*2}"
            configs.append(full_config)

        used_dataset_types.add(dataset_type)

    return configs

# CLI entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate config files for experiments.")
    parser.add_argument("--configname", type=str, default="configs.json", help="Output JSON file name")
    args = parser.parse_args()

    configs = generate_configurations(RUNS_PER_CONFIG)
    with open(os.path.join('configs', args.configname), "w") as f:
        json.dump(configs, f, indent=2)

    print(f"Generated {len(configs)} configurations and saved to {args.configname}")
