from enum import Enum
import os
import random
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Custom Dataset for loading images from either a directory or a list of file paths with labels."""
    
    def __init__(self, image_input, labels=None, transform=None):
        """
        Args:
            image_input (str or list of str): Either a directory path containing images or a list of image file paths.
            labels (list of int or int, optional): List of labels corresponding to the images or a single label for all images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # If input is a directory, load all image paths from the directory
        if isinstance(image_input, str):
            self.image_paths = [os.path.join(image_input, f) for f in os.listdir(image_input) if f.endswith(('.jpeg', '.jpg', '.png'))]
        # If input is a list of image paths
        elif isinstance(image_input, list):
            self.image_paths = image_input
        else:
            raise TypeError("image_input should be either a directory path (str) or a list of file paths (list).")
        
        # Handling labels
        if isinstance(labels, list):
            if len(self.image_paths) != len(labels):
                raise ValueError("Length of image paths and labels must match.")
            self.labels = labels
        elif isinstance(labels, int):
            self.labels = [labels] * len(self.image_paths)  # Apply the same label to all images
        elif labels is None:
            self.labels = [0] * len(self.image_paths)  # Default to 0 if no labels are provided
        else:
            raise TypeError("Labels should be either a list or a single integer value.")
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load the image
        image = Image.open(image_path).convert('RGB')  # Assuming RGB input
        
        # Apply any transformations (e.g., resizing, normalization)
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ProGanDataset(Dataset):
    """
    Dataset for ProGAN with nested directory structure. Loads all images from '0_real' subdirectories.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing class subdirectories with '0_real' and '1_fake'.
            transform (callable, optional): Optional transform to apply to the images.
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Traverse all class folders
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):  # Check if it's a directory
                real_dir = os.path.join(class_path, '0_real')  # Only look for '0_real' subdirectory
                if os.path.isdir(real_dir):  # If '0_real' exists
                    for image_name in os.listdir(real_dir):
                        if image_name.endswith(('.jpeg', '.jpg', '.png')):  # Filter image files
                            self.image_paths.append(os.path.join(real_dir, image_name))
                            self.labels.append(0)  # Assign label 0 for real samples

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')  # Load image and convert to RGB
        if self.transform:
            image = self.transform(image)  # Apply transformations if provided
        return image, label


class CocoDataset(Dataset):
    """
    Dataset class for loading COCO samples listed in the CSV file.
    """

    def __init__(self, root_dir, csv_file="list_train.csv", label=0, transform=None):
        """
        Args:
            root_dir (str): Root directory containing the COCO images (e.g., coco2017/train2017).
            csv_file (str, optional): Path to the CSV file containing image information. Defaults to "train_set/list_train.csv".
            label (int, optional): Label to assign to all samples in this dataset. Defaults to 0 (real samples).
            transform (callable, optional): Optional transform to apply to the images.
        """
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))  # Load CSV file
        self.root_dir = root_dir          # Base directory for the images
        self.transform = transform        # Optional image transformations
        self.label = label                # Fixed label for all samples

        # Extract the file paths for the images
        self.image_paths = self.data['filename0'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the path of the image
        image_path = os.path.join(self.root_dir, self.image_paths[idx])

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, self.label
    

class PatchDataset(Dataset):
    """Dataset that extracts a specific patch from each image in the original dataset."""
    def __init__(self, original_dataset, patch_size, patch_index):
        self.original_dataset = original_dataset
        self.patch_size = patch_size
        self.patch_index = patch_index

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]  # Get the image and label from the original dataset

        # Calculate number of patches along height and width
        h_patches = image.shape[1] // self.patch_size
        w_patches = image.shape[2] // self.patch_size

        # Get row and column for the specific patch index
        row_idx = self.patch_index // w_patches
        col_idx = self.patch_index % w_patches

        # Extract the patch
        patch = image[
            :,  # All channels
            row_idx * self.patch_size:(row_idx + 1) * self.patch_size,
            col_idx * self.patch_size:(col_idx + 1) * self.patch_size
        ]

        return patch, label  # Return the patch and label


def create_inference_dataset(real_dir, fake_dir, num_samples_per_class, classes='both'):
    """
    Create a balanced dataset for inference by sampling images from real and fake directories.
    Args:
        real_dir (str): Directory containing real images.
        fake_dir (str): Directory containing fake images.
        num_samples_per_class (int): Number of samples per class (-1 to load all samples).
        classes (str): Which classes to include ('both', 'real', or 'fake').

    Returns:
        list: List of tuples (image_path, label), where label is 0 for real and 1 for fake.
    """
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]

    # Handle -1 for all samples
    if num_samples_per_class == -1:
        sampled_real_images = real_images
        sampled_fake_images = fake_images
    else:
        sampled_real_images = random.sample(real_images, min(len(real_images), num_samples_per_class))
        sampled_fake_images = random.sample(fake_images, min(len(fake_images), num_samples_per_class))

    # Create dataset of tuples (file_path, label)
    if classes == 'both':
        inference_data = [(img, 0) for img in sampled_real_images] + [(img, 1) for img in sampled_fake_images]
    elif classes == 'fake':
        inference_data = [(img, 1) for img in sampled_fake_images]
    elif classes == 'real':
        inference_data = [(img, 0) for img in sampled_real_images]
    else:
        raise ValueError(f"Invalid classes argument: {classes}. Choose from 'both', 'real', or 'fake'.")

    random.shuffle(inference_data)  # Shuffle the dataset
    return inference_data


class DatasetFactory:
    """Factory class for creating datasets based on dataset type."""
    @staticmethod
    def create_dataset(dataset_type, root_dir, transform=None):
        """
        Create the appropriate dataset based on the dataset type.

        Args:
            dataset_type (str): Type of the dataset ('CelebA', 'ProGan', 'COCO').
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): Transform to apply to the images.

        Returns:
            Dataset: Instance of the appropriate dataset class.
        """
        if dataset_type.upper() == 'CELEBA' or dataset_type.upper() == 'COCO_ALL' or dataset_type.upper() == 'PROGAN_FACES_BUT_CELEBA_AS_TRAIN':
            return ImageDataset(image_input=root_dir, labels=0, transform=transform)
        elif dataset_type.upper() == 'PROGAN':
            return ProGanDataset(root_dir=root_dir, transform=transform)
        elif dataset_type.upper() == 'COCO':
            return CocoDataset(root_dir=root_dir, label=0, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")


# DatasetType Enum
class DatasetType(Enum):
    CELEBA = {
        "data_dir_real": "data/CelebaHQMaskDataset/train/images_faces",
        "data_dir_fake_real": "data/CelebaHQMaskDataset/test/images_faces",
        "data_dir_fake": "data/stable-diffusion-face-dataset/1024/both_faces"
    }
    PROGAN = {
        "data_dir_real": "data/CNNDetector/trainset",
        "data_dir_fake_real": "data/CNNDetector/testset/whichfaceisreal/0_real",
        "data_dir_fake": "data/CNNDetector/testset/whichfaceisreal/1_fake"
    }
    COCO = {
        "data_dir_real": "data/CLIPDetector/train_set/",
        "data_dir_fake_real": "data/CLIPDetector/test_set/real/real_coco_valid",
        "data_dir_fake": "data/CLIPDetector/test_set/fake/sdxl_cocoval"
    }
    COCO_ALL = {
        "data_dir_real": "data/CLIPDetector/train_set/coco2017/train2017",
        "data_dir_fake_real": "data/CLIPDetector/test_set/real/real_coco_valid",
        "data_dir_fake": "data/CLIPDetector/test_set/fake/sdxl_cocoval"
    }
    PROGAN_FACES_BUT_CELEBA_AS_TRAIN = {
        "data_dir_real": "data/CelebaHQMaskDataset/train/images_faces",
        "data_dir_fake_real": "data/CNNDetector/testset/whichfaceisreal/0_real",
        "data_dir_fake": "data/CNNDetector/testset/whichfaceisreal/1_fake"
    }

    def get_paths(self):
        return self.value