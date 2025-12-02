import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import argparse
import numpy as np
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
from datasets_factory import DatasetFactory, DatasetType
from stat_test import generate_combinations, patch_parallel_preprocess
from utils import set_seed, balanced_testset
from torch.utils.data import ConcatDataset
from sklearn.metrics import roc_auc_score, average_precision_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Custom Histogram and Testing Pipeline')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--sample_size', type=int, default=512)
parser.add_argument('--threshold', type=float, default=0.05)
parser.add_argument('--num_samples_per_class', type=int, default=-1)
parser.add_argument('--max_workers', type=int, default=1)
parser.add_argument('--num_data_workers', type=int, default=2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--pkls_dir', type=str, default='pkls/AIStats/rigid_eval')

args = parser.parse_args()


def export_raw_results(dataset_name, seed, image_paths, y_true, y_scores_raw, out_dir="."):
    """
    Save raw per-sample scores (before inversion) with image paths and labels.
    Ensures y_scores_raw is flattened into a 1D list.
    """
    flat_scores = [
        float(s) for sub in y_scores_raw
        for s in (sub if isinstance(sub, (list, np.ndarray)) else [sub])
    ]

    if len(flat_scores) != len(image_paths):
        raise ValueError(f"Length mismatch: {len(flat_scores)} scores vs {len(image_paths)} paths")

    df = pd.DataFrame({
        "image_path": image_paths,
        "score_raw": flat_scores,
        "label": y_true
    })

    out_path = os.path.join(out_dir, f"rigid_dino05_rawscores_{dataset_name}_{seed}.csv")
    df.to_csv(out_path, index=False)
    print(f"Raw scores saved to {out_path}")


def preprocess_and_plot():
    set_seed(args.seed)
    patch_sizes = [args.sample_size]
    waves = ['RIGID.DINO.05']

    dataset_types = ["ALL"]
    results_df = pd.DataFrame(columns=['dataset', 'auc', 'ap'])

    for dataset in tqdm(dataset_types, desc="Datasets", unit="dataset"):
        transform = transforms.Compose([
            transforms.Resize((args.sample_size, args.sample_size)),
            transforms.ToTensor()
        ])

        datasets = DatasetFactory.create_dataset(dataset_type=dataset, transform=transform)
        reference_dataset = datasets['reference_real']
        test_real_dataset = datasets['test_real']
        test_fake_dataset = datasets['test_fake']

        inference_dataset = ConcatDataset([test_real_dataset, test_fake_dataset])
        labels = [0] * len(test_real_dataset) + [1] * len(test_fake_dataset)

        stat_combinations = generate_combinations(patch_sizes, waves)

        real_histograms = patch_parallel_preprocess(
            test_real_dataset, args.batch_size, stat_combinations,
            args.max_workers, args.num_data_workers, args.pkls_dir, seed=args.seed
        )

        fake_histograms = patch_parallel_preprocess(
            test_fake_dataset, args.batch_size, stat_combinations,
            args.max_workers, args.num_data_workers, args.pkls_dir, seed=args.seed
        )

        image_paths = []
        for ds in [test_real_dataset, test_fake_dataset]:
            if hasattr(ds, "image_paths"):
                image_paths.extend(ds.image_paths)
            else:
                raise AttributeError("Dataset is missing attribute 'image_paths'")

        try:
            key = next(iter(real_histograms.keys()))
            if real_histograms[key] is not None and fake_histograms[key] is not None:
                y_true = labels
                y_scores_raw = list(real_histograms[key]) + list(fake_histograms[key])

                export_raw_results(dataset, args.seed, image_paths, y_true, y_scores_raw)

                y_scores = [1 - s for s in y_scores_raw]

                balance_labels, balance_scores = balanced_testset(y_true, y_scores, random_state=42)
                auc = roc_auc_score(balance_labels, balance_scores)
                ap = average_precision_score(balance_labels, balance_scores)

                print(f"[{dataset}] AUC: {auc:.6f}, AP: {ap:.6f}")
                results_df.loc[len(results_df)] = [dataset, auc, ap]

        except Exception as e:
            print(f"Skipped {dataset} due to error: {e}")

    if not results_df.empty:
        output_csv = f'rigid_dino05_results_{args.seed}.csv'
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    else:
        print("No results were saved. Check histogram generation or dataset paths.")


if __name__ == "__main__":
    preprocess_and_plot()
