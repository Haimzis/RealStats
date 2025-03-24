import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from processing.histograms import BaseHistogram


class RIGIDHistogram(BaseHistogram):
    """Computes cosine similarity between original and perturbed images using Hugging Face DINOv2."""

    def __init__(self, model_name="facebook/dinov2-large", noise_level=0.05):
        """
        Args:
            model_name (str): The name of the Hugging Face model to use (default: 'facebook/dinov2-large').
            noise_level (float): Intensity of added Gaussian noise.
        """
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name, do_rescale=False, use_fast=True)
        self.feature_extractor = AutoModel.from_pretrained(model_name).to('cuda').eval()
        self.noise_level = noise_level

    def add_noise(self, image_batch):
        """Applies a single Gaussian noise perturbation to an image batch."""
        noise = torch.randn_like(image_batch) * self.noise_level
        return torch.clamp(image_batch + noise, 0, 1)  # Keep pixel values valid

    def extract_features(self, image_batch):
        """Extracts features from the image batch using the Hugging Face DINOv2 model."""
        with torch.no_grad():
            inputs = self.processor(images=image_batch, return_tensors="pt").to('cuda')
            outputs = self.feature_extractor(**inputs)
            return outputs.last_hidden_state

    def preprocess(self, image_batch):
        """Computes cosine similarity between original and noise-perturbed images."""
        with torch.no_grad():
            original_embedding = self.extract_features(image_batch)
            perturbed_embedding = self.extract_features(self.add_noise(image_batch))
        
        similarity = torch.norm(F.cosine_similarity(original_embedding, perturbed_embedding, dim=-1), p=2, dim=1).cpu().numpy()
        return similarity  # Returns a tensor of cosine similarity scores