import random
import json

# Define parameter ranges
finetune_portion_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
uniform_p_thresholds = [0.05, 0.15, 0.25, 0.4, 0.4, 0.5]
waves_choices = ['bior1.1', 'bior3.1', 'bior6.8', 'coif1', 'coif10', 'db1', 'db38', 'haar', 'rbio6.8', 'sym2', 'fourier', 'dct']
patch_divisors_choices = ["0 1 2", "0 1", "2"]
chi2_bins_choices = [101, 201, 301, 401]
sample_size_choices = [256]
N = 200
# Generate configurations
def generate_configurations(num_configs=N):
    configs = []
    for _ in range(num_configs):
        finetune_portion = random.choice(finetune_portion_range)
        waves = random.sample(waves_choices, random.randint(3, 10))
        patch_divisors = random.choice(patch_divisors_choices)
        uniform_p_threshold = random.choice(uniform_p_thresholds)
        chi2_bins = random.choice(chi2_bins_choices)
        sample_size = 256 if random.random() <= 1.0 else 512  # 90% chance of being 256
        
        config = f"--finetune_portion {finetune_portion} " \
                 f"--waves {' '.join(waves)} " \
                 f"--patch_divisors {patch_divisors} " \
                 f"--chi2_bins {chi2_bins} " \
                 f"--sample_size {sample_size} " \
                 f"--uniform_p_threshold {uniform_p_threshold}"

        configs.append(config)
    return configs

# Generate and save configurations
configs = generate_configurations(N)  # Adjust the number of configurations as needed
with open("configs.json", "w") as f:
    json.dump(configs, f, indent=2)

print(f"Generated {len(configs)} configurations and saved to configs.json")
