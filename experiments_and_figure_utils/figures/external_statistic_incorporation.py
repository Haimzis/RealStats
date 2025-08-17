
import os, argparse, re
from matplotlib import pyplot as plt
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from tqdm import tqdm
import mlflow
from datasets_factory import DatasetType
from stat_test import DataType, TestType, calculate_pvals_from_cdf, generate_combinations, patch_parallel_preprocess, perform_ensemble_testing
from utils import build_backbones_statistics_list, compute_cdf, plot_pvalue_histograms, set_seed
from data_utils import ImageDataset, create_inference_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0 1 2 3"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--sample_size', type=int, default=512)
parser.add_argument('--threshold', type=float, default=0.05)
parser.add_argument('--dataset_type', type=str, default='COCO_ALL', choices=[e.name for e in DatasetType])
parser.add_argument('--num_samples_per_class', type=int, default=-1)
parser.add_argument('--max_workers', type=int, default=4)
parser.add_argument('--num_data_workers', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--pkls_dir', type=str, default='pkls_adaptability')
parser.add_argument('--mlflow_experiment', type=str, default='R minp no patch dino clip low - Experiments II - With AP')
args = parser.parse_args()

histograms_stats_dir = os.path.join('histograms_stats', '29.07_adaptability')

def evaluate_independent_key_runs(real_histograms, fake_histograms, real_paths, fake_paths, hist_dir, dataset, experiment_name, threshold, filter_out_bias=False):
    auc_list = []
    mlflow.set_experiment(experiment_name)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp: return
    for _, r in mlflow.search_runs(exp.experiment_id).iterrows():
        run_id = r["run_id"]
        run_name = r.get("tags.mlflow.runName", run_id)
        try:
            run = client.get_run(run_id)
            params = run.data.params
            if "dataset_type" not in params or params["dataset_type"] != dataset: 
                continue

            raw_keys = eval(params["Independent keys"])
            keys = [re.sub(r"seed=\d+", f"seed={args.seed}", k) for k in raw_keys] + [k for k in real_histograms.keys() if 'ManifoldBias' in k]
            ensemble = params.get("ensemble_test", "minp")
            # ensemble = 'manual-stouffer'
            real_sel = {k: [h.squeeze() for h in real_histograms[k]] for k in keys if k in real_histograms}
            fake_sel = {k: [h.squeeze() for h in fake_histograms[k]] for k in keys if k in fake_histograms}
            if not real_sel or not fake_sel: 
                continue

            cdfs = {k: compute_cdf(real_sel[k], bins=500, test_id=k) for k in keys}
            inf_hist = {k: real_sel[k] + fake_sel[k] for k in keys}
            pvals = np.clip(calculate_pvals_from_cdf(cdfs, inf_hist, DataType.TEST.name, TestType.BOTH), 0, 1)
            _, ensemble_pvals = perform_ensemble_testing(pvals, ensemble_test=ensemble, plot=False)

            labels = [0] * len(real_paths) + [1] * len(fake_paths)
            auc = roc_auc_score(labels, 1 - np.array(ensemble_pvals))
            auc_list.append(auc)

            suffix = "before" if filter_out_bias else "after"
            out = os.path.join(hist_dir, dataset, run_name)
            os.makedirs(out, exist_ok=True)
            plot_pvalue_histograms(
                [p for p, l in zip(ensemble_pvals, labels) if l == 0],
                [p for p, l in zip(ensemble_pvals, labels) if l == 1],
                os.path.join(out, f"{ensemble}_pval_plot_{suffix}.svg"),
                f"P-values on {dataset.replace('_TEST_ONLY', '')} ({suffix.replace('_',' ')})",
                bins=30,
                figsize=(6, 6),
                title_fontsize=16, label_fontsize=14, legend_fontsize=12
            )
        
        except Exception as e:
            print(f"Skipping run {run_name}: {e}")
    
    return np.mean(auc_list)

def preprocess_and_plot():
    set_seed(args.seed)
    patch_sizes = [args.sample_size]
    levels = [0]
    waves = build_backbones_statistics_list(['DINO','BEIT','CLIP','RESNET'], ['01','05','10'])
    # waves = build_backbones_statistics_list(['DINO'], ['01','05','10'])
    dataset_types = [
        # 'COCO_STABLE_DIFFUSION_1_4_TEST_ONLY', 
        'GAUGAN_TEST_ONLY',
        'SEEINGDARK_TEST_ONLY',
        'COCO_STABLE_DIFFUSION_2_TEST_ONLY'
        ]

    for dataset in tqdm(dataset_types, desc="Datasets"):
        print('DATASET: ', dataset)
        out_dir = os.path.join(histograms_stats_dir, dataset)
        pkl_dir = os.path.join(args.pkls_dir, dataset)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(pkl_dir, exist_ok=True)
        transform = transforms.Compose([transforms.Resize((args.sample_size, args.sample_size)), transforms.ToTensor()])
        paths = DatasetType[dataset.upper()].get_paths()
        inference_data = create_inference_dataset(paths['test_real']['path'], paths['test_fake']['path'], args.num_samples_per_class, classes='both')
        real_paths = [x[0] for x in inference_data if x[1] == 0]
        fake_paths = [x[0] for x in inference_data if x[1] == 1]
        real_ds = ImageDataset(real_paths, [0]*len(real_paths), transform)
        fake_ds = ImageDataset(fake_paths, [1]*len(fake_paths), transform)
        stat_combs = generate_combinations(patch_sizes, waves, levels)

        real_hist = patch_parallel_preprocess(real_ds, args.batch_size, stat_combs, args.max_workers, args.num_data_workers, pkl_dir, True, DataType.CALIB)
        fake_hist = patch_parallel_preprocess(fake_ds, args.batch_size, stat_combs, args.max_workers, args.num_data_workers, pkl_dir, True, DataType.TEST)

        real_hist_before, fake_hist_before = real_hist.copy(), fake_hist.copy()

        # ManifoldBias
        df = pd.read_csv(os.path.join(args.pkls_dir, f'{dataset}.csv'))
        real_sorted = df[df.label == 0].set_index('image_path').loc[real_paths].reset_index()
        fake_sorted = df[df.label == 1].set_index('image_path').loc[fake_paths].reset_index()
        key = f'PatchProcessing_wavelet=ManifoldBias_level=0_patch_size=512_seed={args.seed}'
        real_hist[key] = np.array([[v] for v in real_sorted['criterion'].tolist()])
        fake_hist[key] = np.array([[v] for v in fake_sorted['criterion'].tolist()])

        average_mean = evaluate_independent_key_runs(real_hist_before, fake_hist_before, real_paths, fake_paths, histograms_stats_dir, dataset, args.mlflow_experiment, args.threshold, filter_out_bias=True)
        average_mean_with = evaluate_independent_key_runs(real_hist, fake_hist, real_paths, fake_paths, histograms_stats_dir, dataset, args.mlflow_experiment, args.threshold, filter_out_bias=False)
        print(f"Average AUC before ManifoldBias: {average_mean}, after ManifoldBias: {average_mean_with}")
        
if __name__ == "__main__":
    preprocess_and_plot()
