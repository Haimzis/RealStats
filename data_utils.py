import os
import random
from PIL import Image
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
            self.image_paths = [os.path.join(image_input, f) for f in os.listdir(image_input) if f.endswith(('.jpg', '.png'))]
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
        h_patches = image.shape[1] // self.patch_size[0]
        w_patches = image.shape[2] // self.patch_size[1]

        # Get row and column for the specific patch index
        row_idx = self.patch_index // w_patches
        col_idx = self.patch_index % w_patches

        # Extract the patch
        patch = image[
            :,  # All channels
            row_idx * self.patch_size[0]:(row_idx + 1) * self.patch_size[0],
            col_idx * self.patch_size[1]:(col_idx + 1) * self.patch_size[1]
        ]

        return patch, label  # Return the patch and label


def create_inference_dataset(real_dir, fake_dir, num_samples_per_class, classes='both'):
    """Create a balanced dataset for inference by sampling images from real and fake directories."""
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))]

    # Create dataset of tuples (file_path, label)
    if classes == 'both':
        sampled_real_images = random.sample(real_images, num_samples_per_class)
        sampled_fake_images = random.sample(fake_images, num_samples_per_class)
        inference_data = [(img, 0) for img in sampled_real_images] + [(img, 1) for img in sampled_fake_images]
    elif classes == 'fake':
        sampled_fake_images = random.sample(fake_images, num_samples_per_class)
        inference_data = [(img, 1) for img in sampled_fake_images]
    else: # Real
        sampled_real_images = random.sample(real_images, num_samples_per_class)
        inference_data = [(img, 0) for img in sampled_real_images] 

    random.shuffle(inference_data)  # Shuffle the dataset
    return inference_data