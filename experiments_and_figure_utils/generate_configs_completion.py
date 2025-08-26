import os
import random
import json
import argparse

from utils import build_backbones_statistics_list

# Define parameter ranges
finetune_portion_range = [0.1]  # treated as list for consistency
dataset_types = [
    "COCO_STABLE_DIFFUSION_1_4_TEST_ONLY",
    "IMAGENET_GLIDE_100_10",
    "IMAGENET_GLIDE_100_27",
    "IMAGENET_LDM",
    "LAION_GLIDE_100_10",
    "LAION_GLIDE_100_27",
    "LAION_LDM",
    "PROGAN_GLIDE_100_10",
    "PROGAN_GLIDE_100_27",
    "PROGAN_LDM"
]

# models = ['DINO', 'BEIT', 'CLIP', 'DEIT', 'RESNET']
# noise_levels = ['01', '05', '10', '50', '75', '100']

models_1 = ['DINO', 'CLIP']
noise_levels_1 = ['01', '05', '10']

models_2 = ['RESNET']
noise_levels_2 = ['05', '10']

models_3 = ['BEIT']
noise_levels_3 = ['05', '10']

statistics_choices = []
statistics_choices += build_backbones_statistics_list(models_1, noise_levels_1)
statistics_choices += build_backbones_statistics_list(models_2, noise_levels_2)
statistics_choices += build_backbones_statistics_list(models_3, noise_levels_3)

# patch_divisors_choices = ["0 1", "0 1", "0", "0", "0"]
patch_divisors_choices = ["0"]
chi2_bins_choices = [20]
statistic_ensemble = "minp"  # minp

N = 100
RUNS_PER_CONFIG = 15  # generate each config with 5 different seeds

def generate_configurations(num_configs, runs_per_config):
    configs = []
    used_dataset_types = set()

    # Force each dataset_type to appear at least once
    for dataset_type in dataset_types:
        finetune_portion = random.choice(finetune_portion_range)
        statistics = statistics_choices
        patch_divisors = random.choice(patch_divisors_choices)
        chi2_bins = random.choice(chi2_bins_choices)

        base_config = (
            f"--finetune_portion {finetune_portion} "
            f"--statistics {' '.join(statistics)} "
            f"--ensemble_test {statistic_ensemble} "
            f"--patch_divisors {patch_divisors} "
            f"--chi2_bins {chi2_bins} "
            f"--dataset_type {dataset_type}"
        )

        for _ in range(runs_per_config):
            seed = random.randint(0, 100000)
            full_config = f"{base_config} --seed {seed}"
            configs.append(full_config)

        used_dataset_types.add(dataset_type)

    # # Fill the rest randomly if needed
    # remaining_configs = num_configs - len(dataset_types)
    # for _ in range(remaining_configs):
    #     dataset_type = random.choice(dataset_types)
    #     finetune_portion = random.choice(finetune_portion_range)
    #     statistics = random.sample(statistics_choices, 15)
    #     patch_divisors = random.choice(patch_divisors_choices)
    #     chi2_bins = random.choice(chi2_bins_choices)

    #     base_config = (
    #         f"--finetune_portion {finetune_portion} "
    #         f"--statistics {' '.join(statistics)} "
    #         f"--patch_divisors {patch_divisors} "
    #         f"--chi2_bins {chi2_bins} "
    #         f"--dataset_type {dataset_type}"
    #     )

    #     for _ in range(runs_per_config):
    #         seed = random.randint(0, 100000)
    #         full_config = f"{base_config} --seed {seed}"
    #         configs.append(full_config)

    return configs

# CLI entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate config files for experiments.")
    parser.add_argument("--configname", type=str, default="configs.json", help="Output JSON file name")
    args = parser.parse_args()

    configs = generate_configurations(N, RUNS_PER_CONFIG)
    with open(os.path.join('configs', args.configname), "w") as f:
        json.dump(configs, f, indent=2)

    print(f"Generated {len(configs)} configurations and saved to {args.configname}")
