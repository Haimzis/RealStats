import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from processing.histograms import BaseHistogram
from transformers import AutoImageProcessor, CLIPModel
from transformers import AutoImageProcessor, ResNetForImageClassification

class RIGIDHistogram(BaseHistogram):
    def __init__(self, model_name, noise_level=0.05):
        super().__init__()
        self.model_name = model_name
        self.noise_level = noise_level
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = self.load_processor()
        self.model = self.load_model().to(self.device).eval()

    def load_processor(self):
        """Must be implemented by subclass."""
        raise NotImplementedError

    def load_model(self):
        """Must be implemented by subclass."""
        raise NotImplementedError

    def extract_features(self, image_batch):
        """Extract features from the image batch."""
        with torch.no_grad():
            inputs = self.processor(images=image_batch, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            return self.get_embedding(outputs)

    def get_embedding(self, outputs):
        """Must be implemented by subclass. Extracts embedding tensor from model outputs."""
        raise NotImplementedError

    def add_noise(self, image_batch):
        noise = torch.randn_like(image_batch) * self.noise_level
        return torch.clamp(image_batch + noise, 0, 1)

    def preprocess(self, image_batch):
        with torch.no_grad():
            original = self.extract_features(image_batch)
            perturbed = self.extract_features(self.add_noise(image_batch))

        similarity = torch.norm(F.cosine_similarity(original, perturbed, dim=-1), p=2, dim=1).cpu().numpy()
        return similarity


class RIGIDDinoV2Histogram(RIGIDHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="facebook/dinov2-large", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        return AutoModel.from_pretrained(self.model_name)

    def get_embedding(self, outputs):
        return outputs.last_hidden_state  # shape: [B, seq_len, dim]
    
class RIGIDEVAHistogram(RIGIDHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="eva_clip_g/eva02-large-336", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        return AutoModel.from_pretrained(self.model_name)

    def get_embedding(self, outputs):
        return outputs.last_hidden_state  # [B, tokens, dim]

class RIGIDBEiTHistogram(RIGIDHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="microsoft/beit-large-patch16-224", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        return AutoModel.from_pretrained(self.model_name)

    def get_embedding(self, outputs):
        return outputs.last_hidden_state


class RIGIDOpenCLIPHistogram(RIGIDHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        model = CLIPModel.from_pretrained(self.model_name)
        return model.vision_model  # ✅ Use only the vision tower

    def get_embedding(self, outputs):
        return outputs.last_hidden_state  # [B, seq_len, dim]

    
class RIGIDConvNeXtHistogram(RIGIDHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="facebook/convnext-base-224", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        return AutoModel.from_pretrained(self.model_name)

    def get_embedding(self, outputs):
        # outputs.last_hidden_state: [B, C, H, W]
        x = outputs.last_hidden_state
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # → [B, C, 1, 1]
        x = x.view(x.size(0), 1, -1)  # → [B, 1, C]
        return x


class RIGIDResNet50Histogram(RIGIDHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="microsoft/resnet-50", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        return AutoModel.from_pretrained(self.model_name)  # Loads ResNetModel (no classifier head)

    def get_embedding(self, outputs):
        # outputs.last_hidden_state shape: [B, C, H, W]
        x = outputs.last_hidden_state
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # [B, C, 1, 1]
        x = x.view(x.size(0), 1, -1)  # [B, 1, C]
        return x