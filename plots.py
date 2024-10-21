from concurrent.futures import ThreadPoolExecutor
import random
import pywt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from data_utils import ImageDataset
from processing.histograms import EnergyHistogram, FourierEnergyHistogram, NormHistogram
from processing.pca import PCAOutlierDetector
from processing.tsne import TSNEWavelet
from utils import set_seed

def main(data_dir_real, data_dir_fake, batch_size=128, action='pca', figname='pca.png', wavelet='haar', selected_indices=[0]):
    # Initialize datasets and transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    real_dataset = ImageDataset(data_dir_real, transform=transform, labels=0)
    fake_dataset = ImageDataset(data_dir_fake, transform=transform, labels=1)

    # Determine which pipeline to run based on the action
    if action == 'histogram':
        histogram_pipeline(real_dataset, fake_dataset, batch_size, figname, wavelet, selected_indices)
    elif action == 'tsne':
        tsne_pipeline(real_dataset, fake_dataset, batch_size, figname, wavelet, selected_indices)
    elif action == 'pca':
        pca_pipeline(real_dataset, fake_dataset, batch_size, figname, wavelet, selected_indices)
    else:
        print("Invalid input.")
    
    print("Process completed.")

def histogram_pipeline(real_dataset, fake_dataset, batch_size, figname, wavelet, selected_indices=[0]):
    # Initialize data loaders
    data_loader_real = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    data_loader_fake = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Generate histograms with the specified wavelet
    # histogram_generator = EnergyHistogram(J=7, selected_indices=[6], wavelet=wavelet)
    # histogram_generator = FourierEnergyHistogram(wavelet=wavelet)
    histogram_generator = NormHistogram(selected_indices=selected_indices, wave=wavelet)
    histograms_real = histogram_generator.create_histogram(data_loader_real)
    histograms_fake = histogram_generator.create_histogram(data_loader_fake)

    # Plot histograms
    histogram_generator.plot_histograms(histograms_real, histograms_fake, figname)

def tsne_pipeline(real_dataset, fake_dataset, batch_size, figname, wavelet, selected_indices=[0]):
    # Create data loaders for real and fake datasets
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    real_subset_dataset = Subset(real_dataset, list(range(len(fake_dataset))))
    real_subset_loader = DataLoader(real_subset_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize t-SNE generator and run t-SNE with the specified wavelet
    tsne_generator = TSNEWavelet(J=7, selected_indices=selected_indices, wave=wavelet)
    tsne_generator.run_tsne(real_subset_loader, fake_loader, figname)

def pca_pipeline(real_dataset, fake_dataset, batch_size, figname, wavelet, selected_indices=[0]):
    # Create data loaders for real and fake datasets
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize PCA Outlier Detector and run PCA with the specified wavelet
    pca_detector = PCAOutlierDetector(J=7, selected_indices=selected_indices, wave=wavelet)
    pca_detector.run_pca_outlier_detection(real_loader, fake_loader, figname)

def process_wavelet_task(params, device_id):
    data_dir_real, data_dir_fake, batch_size, action, wavelet, level = params
    figname = f'norm_hist_hf{level}_wave_{wavelet}.png'
    selected_indices = [level]

    # Assign the specific GPU device for this process
    torch.cuda.set_device(device_id)
    print(f"Using GPU: {device_id} for wavelet {wavelet} at level {level}")

    main(data_dir_real, data_dir_fake, batch_size=batch_size, action=action, figname=figname, wavelet=wavelet, selected_indices=selected_indices)

if __name__ == "__main__":
    set_seed(42)
    data_dir_real = 'data/real/train'
    data_dir_fake = 'data/fake'
    action = 'histogram'
    
    # Define the wavelet families, wavelets, and levels
    wavelet_tasks = []
    for wave_family in ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']: 
        for wavelet in pywt.wavelist(wave_family):
            for level in [0, 2, 4, 6]:
                wavelet_tasks.append((data_dir_real, data_dir_fake, 256, action, wavelet, level))

    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    
    # Create ThreadPoolExecutor to run tasks in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        for task in wavelet_tasks:
            # Randomly assign a GPU for each task
            random_gpu = random.randint(0, num_gpus - 1)
            executor.submit(process_wavelet_task, task, random_gpu)