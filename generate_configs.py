import random
import json

# Define parameter ranges
finetune_portion_range = [0.1, 0.15, 0.2, 0.25]
dataset_types = [
    'PROGAN_FACES', 'COCO', 'COCO_BIGGAN_256', 'COCO_STABLE_DIFFUSION_XL', 
    'COCO_DALLE3_COCOVAL', 'COCO_SYNTH_MIDJOURNEY_V5', 'COCO_STABLE_DIFFUSION_2',
    'BIGGAN_TEST_ONLY', 'CYCLEGAN_TEST_ONLY', 'GAUGAN_TEST_ONLY', 'PROGAN_TEST_ONLY',
    'SEEINGDARK_TEST_ONLY', 'STYLEGAN_TEST_ONLY', 'CRN_TEST_ONLY', 'DEEPFAKE_TEST_ONLY',
    'IMLE_TEST_ONLY', 'SAN_TEST_ONLY', 'STARGAN_TEST_ONLY', 'STYLEGAN2_TEST_ONLY', 'COCO_STABLE_DIFFUSION_2_768'
]
waves_choices = \
[
            # DINOv2
            'RIGID.DINO.001', 'RIGID.DINO.01', 'RIGID.DINO.05', 'RIGID.DINO.10', 'RIGID.DINO.20', 'RIGID.DINO.30', 'RIGID.DINO.50',
            # BEiT
            'RIGID.BEIT.01', 'RIGID.BEIT.05', 'RIGID.BEIT.10', 'RIGID.BEIT.20', 'RIGID.BEIT.30', 'RIGID.BEIT.50',
            # OpenCLIP
            'RIGID.CLIP.001', 'RIGID.CLIP.01', 'RIGID.CLIP.05', 'RIGID.CLIP.10', 'RIGID.CLIP.20', 'RIGID.CLIP.30', 'RIGID.CLIP.50',
            # DeiT
            'RIGID.DEIT.05', 'RIGID.DEIT.10', 'RIGID.DEIT.20', 'RIGID.DEIT.30', 'RIGID.DEIT.50',
            # ResNet
            'RIGID.RESNET.05', 'RIGID.RESNET.10', 'RIGID.RESNET.20', 'RIGID.RESNET.30', 'RIGID.RESNET.50'
]
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
