import os
import shutil
import csv
import numpy as np
import random
import string
from PIL import Image

def generate_noise_image(path, size=(256, 256)):
    """Generate a random noise image and save it."""
    noise = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(noise, 'RGB')
    img.save(path)

def random_dir_name(length=8):
    """Generate a random alphanumeric directory name."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def setup_directories(base_dir, num_datasets=4, num_splits=3):
    """Create dataset directories with random sub-split names."""
    dataset_paths = {}
    for i in range(num_datasets):
        dataset_name = f"A_{i}"
        dataset_paths[dataset_name] = []
        for _ in range(num_splits):
            dir_name = random_dir_name()
            dir_path = os.path.join(base_dir, dataset_name, dir_name, "0_real")
            os.makedirs(dir_path, exist_ok=True)
            dataset_paths[dataset_name].append((dir_name, dir_path))
    return dataset_paths

def copy_images_to_directories(image_path, dataset_paths, copies=64):
    """Copy the given image multiple times into each dataset directory."""
    all_entries = []
    for dataset_name, split_dirs in dataset_paths.items():
        for dir_name, split_dir in split_dirs:
            for j in range(copies):
                filename = f"{j}.png"
                target_path = os.path.join(split_dir, filename)
                shutil.copy(image_path, target_path)
                # Correct relative path: include random dir + 0_real
                rel_path = os.path.join(dir_name, "0_real", filename)
                all_entries.append((dataset_name, rel_path))
    return all_entries

def create_csvs(base_dir, entries):
    """Create CSV files with dataset_name and relative paths."""
    csv_files = [
        "test_generated_paths.csv",
        "test_real_paths.csv",
        "reference_paths.csv",
    ]

    for csv_name in csv_files:
        csv_path = os.path.join(base_dir, csv_name)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset_name", "path"])
            writer.writerows(entries)

def main():
    base_dir = os.path.join("data", "ManifoldBias")
    os.makedirs(base_dir, exist_ok=True)

    # Step 1: Generate base noise image
    noise_img_path = os.path.join(base_dir, "a.png")
    generate_noise_image(noise_img_path)

    # Step 2: Setup directories with random names
    dataset_paths = setup_directories(base_dir)

    # Step 3: Copy images + collect correct relative paths
    entries = copy_images_to_directories(noise_img_path, dataset_paths)

    # Step 4: Create CSVs with correct paths
    create_csvs(base_dir, entries)

    print(f"Dataset generated successfully under {base_dir}")

if __name__ == "__main__":
    main()
