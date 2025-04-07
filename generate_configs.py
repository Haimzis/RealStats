import random
import json

from utils import build_backbones_statistics_list

# Define parameter ranges
finetune_portion_range = [0.1, 0.15, 0.2, 0.25]
dataset_types = [
    'PROGAN_FACES', 'COCO', 'COCO_BIGGAN_256', 'COCO_STABLE_DIFFUSION_XL', 
    'COCO_DALLE3_COCOVAL', 'COCO_SYNTH_MIDJOURNEY_V5', 'COCO_STABLE_DIFFUSION_2',
    'BIGGAN_TEST_ONLY', 'CYCLEGAN_TEST_ONLY', 'GAUGAN_TEST_ONLY', 'PROGAN_TEST_ONLY',
    'SEEINGDARK_TEST_ONLY', 'STYLEGAN_TEST_ONLY', 'CRN_TEST_ONLY', 'DEEPFAKE_TEST_ONLY',
    'IMLE_TEST_ONLY', 'SAN_TEST_ONLY', 'STARGAN_TEST_ONLY', 'STYLEGAN2_TEST_ONLY', 'COCO_STABLE_DIFFUSION_2_768'
]

models = ['CONVNEXT', 'DINO', 'BEIT', 'CLIP', 'DEIT', 'RESNET']
noise_levels = ['01', '05', '10', '50', '75', '100']

waves_choices = build_backbones_statistics_list(models, noise_levels)
    
patch_divisors_choices = ["0 1", "0 1", "0", "0", "0"]
chi2_bins_choices = [5, 10]
N = 200

# Generate configurations
def generate_configurations(num_configs=N):
    configs = []
    for _ in range(num_configs):
        finetune_portion = random.choice(finetune_portion_range)
        waves = random.sample(waves_choices, random.randint(4, 10))
        patch_divisors = random.choice(patch_divisors_choices)
        chi2_bins = random.choice(chi2_bins_choices)
        dataset_type = random.choice(dataset_types)

        config = f"--finetune_portion {finetune_portion} " \
                 f"--waves {' '.join(waves)} " \
                 f"--patch_divisors {patch_divisors} " \
                 f"--chi2_bins {chi2_bins} " \
                 f"--dataset_type {dataset_type} " \

        configs.append(config)
    return configs

# Generate and save configurations
configs = generate_configurations(N)  # Adjust the number of configurations as needed
with open("configs.json", "w") as f:
    json.dump(configs, f, indent=2)

print(f"Generated {len(configs)} configurations and saved to configs.json")
