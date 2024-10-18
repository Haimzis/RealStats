import os
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
