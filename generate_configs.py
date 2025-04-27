import random
import json
import argparse

from utils import build_backbones_statistics_list

# Define parameter ranges
finetune_portion_range = [0.1]  # treated as list for consistency
dataset_types = [
    'BIGGAN_TEST_ONLY', 'CYCLEGAN_TEST_ONLY', 'GAUGAN_TEST_ONLY', 'PROGAN_TEST_ONLY',
    'SEEINGDARK_TEST_ONLY', 'STYLEGAN_TEST_ONLY', 'CRN_TEST_ONLY', 'DEEPFAKE_TEST_ONLY',
    'IMLE_TEST_ONLY', 'SAN_TEST_ONLY', 'STARGAN_TEST_ONLY', 'STYLEGAN2_TEST_ONLY',
    'CELEBA_TEST_ONLY', 'COCO_TEST_ONLY', 'COCO_BIGGAN_256_TEST_ONLY',
    'COCO_STABLE_DIFFUSION_XL_TEST_ONLY', 'COCO_DALLE3_COCOVAL_TEST_ONLY',
    'COCO_SYNTH_MIDJOURNEY_V5_TEST_ONLY', 'COCO_STABLE_DIFFUSION_2_TEST_ONLY'
]

models = ['DINO', 'BEIT', 'CLIP', 'DEIT', 'RESNET']
noise_levels = ['01', '05', '10', '50', '75', '100']

waves_choices = build_backbones_statistics_list(models, noise_levels)
patch_divisors_choices = ["0 1", "0 1", "0", "0", "0"]
chi2_bins_choices = [5, 10]

N = 100
RUNS_PER_CONFIG = 5  # generate each config with 5 different seeds

def generate_configurations(num_configs, runs_per_config):
    configs = []
    used_dataset_types = set()

    # Force each dataset_type to appear at least once
    for dataset_type in dataset_types:
        finetune_portion = random.choice(finetune_portion_range)
        waves = random.sample(waves_choices, 15)
        patch_divisors = random.choice(patch_divisors_choices)
        chi2_bins = random.choice(chi2_bins_choices)

        base_config = (
            f"--finetune_portion {finetune_portion} "
            f"--waves {' '.join(waves)} "
            f"--patch_divisors {patch_divisors} "
            f"--chi2_bins {chi2_bins} "
            f"--dataset_type {dataset_type}"
        )

        for _ in range(runs_per_config):
            seed = random.randint(0, 100000)
            full_config = f"{base_config} --seed {seed}"
            configs.append(full_config)

        used_dataset_types.add(dataset_type)

    # Fill the rest randomly if needed
    remaining_configs = num_configs - len(dataset_types)
    for _ in range(remaining_configs):
        dataset_type = random.choice(dataset_types)
        finetune_portion = random.choice(finetune_portion_range)
        waves = random.sample(waves_choices, 15)
        patch_divisors = random.choice(patch_divisors_choices)
        chi2_bins = random.choice(chi2_bins_choices)

        base_config = (
            f"--finetune_portion {finetune_portion} "
            f"--waves {' '.join(waves)} "
            f"--patch_divisors {patch_divisors} "
            f"--chi2_bins {chi2_bins} "
            f"--dataset_type {dataset_type}"
        )

        for _ in range(runs_per_config):
            seed = random.randint(0, 100000)
            full_config = f"{base_config} --seed {seed}"
            configs.append(full_config)

    return configs

# CLI entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate config files for experiments.")
    parser.add_argument("--configname", type=str, default="configs.json", help="Output JSON file name")
    args = parser.parse_args()

    configs = generate_configurations(N, RUNS_PER_CONFIG)
    with open(args.configname, "w") as f:
        json.dump(configs, f, indent=2)

    print(f"Generated {len(configs)} configurations and saved to {args.configname}")
