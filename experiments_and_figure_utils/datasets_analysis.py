from enum import Enum
from data_utils import CocoDataset, ImageDataset, ProGanDataset
from datasets_factory import DatasetType, DatasetFactory  # adjust import as needed

def print_dataset_sample_counts():
    print("Sample counts (excluding PROGAN datasets):\n")
    for name, member in DatasetType.__members__.items():
        if "PROGAN" in name:
            continue  # Skip PROGAN-related datasets

        try:
            dataset_info = member.get_paths()
            counts = {}

            for split, split_info in dataset_info.items():
                dataset_class = split_info["class"]
                dataset_path = split_info["path"]

                # Instantiate dataset (label is dummy: 0 for real, 1 for fake)
                label = 0 if "real" in split else 1
                dataset = dataset_class(dataset_path, label)

                counts[split] = len(dataset)

            print(f"Dataset: {name}")
            print(f"  Train Real: {counts.get('train_real', 0):>5}  |  Train Fake: {counts.get('train_fake', 0):>5}")
            print(f"  Test  Real: {counts.get('test_real', 0):>5}  |  Test  Fake: {counts.get('test_fake', 0):>5}")
            print("-" * 60)

        except Exception as e:
            print(f"  [!] Failed to process {name}: {e}")
            print("-" * 60)

if __name__ == "__main__":
    print_dataset_sample_counts()
