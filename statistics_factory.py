from typing import Optional

from processing.manifold_bias_histogram import LatentNoiseCriterionOriginal
from processing.rigid_histogram import RIGIDBEiTHistogram, RIGIDBigGCLIPHistogram, RIGIDConvNeXtHistogram, RIGIDDinoV2Histogram, RIGIDDinoV3Histogram, RIGIDEVAHistogram, RIGIDOpenAICLIPHistogram, RIGIDOpenCLIPHistogram, RIGIDResNet50Histogram


STATISTIC_HISTOGRAMS = {
    # Paper Statistics
    'LatentNoiseCriterion_original': lambda: LatentNoiseCriterionOriginal(),

    # ==============================
    #    Hardcoded RIGID Statistics
    # ==============================
    
    # ===== DINO =====
    'RIGID.DINO.001': lambda: RIGIDDinoV2Histogram(noise_level=0.001),
    'RIGID.DINO.01': lambda: RIGIDDinoV2Histogram(noise_level=0.01),
    'RIGID.DINO.05': lambda: RIGIDDinoV2Histogram(noise_level=0.05),
    'RIGID.DINO.10': lambda: RIGIDDinoV2Histogram(noise_level=0.10),
    'RIGID.DINO.20': lambda: RIGIDDinoV2Histogram(noise_level=0.20),
    'RIGID.DINO.30': lambda: RIGIDDinoV2Histogram(noise_level=0.30),
    'RIGID.DINO.50': lambda: RIGIDDinoV2Histogram(noise_level=0.50),
    'RIGID.DINO.75': lambda: RIGIDDinoV2Histogram(noise_level=0.75),
    'RIGID.DINO.100': lambda: RIGIDDinoV2Histogram(noise_level=1.0),

    # ===== DINOv3 =====
    # --- ViT-H/16 ---
    'RIGID.DINOV3.VITH16.001': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.001),
    'RIGID.DINOV3.VITH16.01': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.01),
    'RIGID.DINOV3.VITH16.05': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.05),
    'RIGID.DINOV3.VITH16.10': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.10),
    'RIGID.DINOV3.VITH16.20': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.20),
    'RIGID.DINOV3.VITH16.30': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.30),
    'RIGID.DINOV3.VITH16.50': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.50),
    'RIGID.DINOV3.VITH16.75': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.75),
    'RIGID.DINOV3.VITH16.100': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=1.0),

    # --- ViT-S/16 ---
    'RIGID.DINOV3.VITS16.001': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.001),
    'RIGID.DINOV3.VITS16.01': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.01),
    'RIGID.DINOV3.VITS16.05': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.05),
    'RIGID.DINOV3.VITS16.10': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.10),
    'RIGID.DINOV3.VITS16.20': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.20),
    'RIGID.DINOV3.VITS16.30': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.30),
    'RIGID.DINOV3.VITS16.50': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.50),
    'RIGID.DINOV3.VITS16.75': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.75),
    'RIGID.DINOV3.VITS16.100': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=1.0),

    # --- ConvNeXt-Small ---
    'RIGID.DINOV3.CONVNEXTSMALL.001': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.001),
    'RIGID.DINOV3.CONVNEXTSMALL.01': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.01),
    'RIGID.DINOV3.CONVNEXTSMALL.05': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.05),
    'RIGID.DINOV3.CONVNEXTSMALL.10': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.10),
    'RIGID.DINOV3.CONVNEXTSMALL.20': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.20),
    'RIGID.DINOV3.CONVNEXTSMALL.30': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.30),
    'RIGID.DINOV3.CONVNEXTSMALL.50': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.50),
    'RIGID.DINOV3.CONVNEXTSMALL.75': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.75),
    'RIGID.DINOV3.CONVNEXTSMALL.100': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=1.0),

    # ===== BEiT =====
    'RIGID.BEIT.001': lambda: RIGIDBEiTHistogram(noise_level=0.001),
    'RIGID.BEIT.01': lambda: RIGIDBEiTHistogram(noise_level=0.01),
    'RIGID.BEIT.05': lambda: RIGIDBEiTHistogram(noise_level=0.05),
    'RIGID.BEIT.10': lambda: RIGIDBEiTHistogram(noise_level=0.10),
    'RIGID.BEIT.20': lambda: RIGIDBEiTHistogram(noise_level=0.20),
    'RIGID.BEIT.30': lambda: RIGIDBEiTHistogram(noise_level=0.30),
    'RIGID.BEIT.50': lambda: RIGIDBEiTHistogram(noise_level=0.50),
    'RIGID.BEIT.75': lambda: RIGIDBEiTHistogram(noise_level=0.75),
    'RIGID.BEIT.100': lambda: RIGIDBEiTHistogram(noise_level=1.0),

    # ===== OpenCLIP ViT-H =====
    'RIGID.CLIP.001': lambda: RIGIDOpenCLIPHistogram(noise_level=0.001),
    'RIGID.CLIP.01': lambda: RIGIDOpenCLIPHistogram(noise_level=0.01),
    'RIGID.CLIP.05': lambda: RIGIDOpenCLIPHistogram(noise_level=0.05),
    'RIGID.CLIP.10': lambda: RIGIDOpenCLIPHistogram(noise_level=0.10),
    'RIGID.CLIP.20': lambda: RIGIDOpenCLIPHistogram(noise_level=0.20),
    'RIGID.CLIP.30': lambda: RIGIDOpenCLIPHistogram(noise_level=0.30),
    'RIGID.CLIP.50': lambda: RIGIDOpenCLIPHistogram(noise_level=0.50),
    'RIGID.CLIP.75': lambda: RIGIDOpenCLIPHistogram(noise_level=0.75),
    'RIGID.CLIP.100': lambda: RIGIDOpenCLIPHistogram(noise_level=1.0),

    # ===== OpenAI CLIP ViT-L/14 =====
    'RIGID.CLIPOPENAI.001': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.001),
    'RIGID.CLIPOPENAI.01': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.01),
    'RIGID.CLIPOPENAI.05': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.05),
    'RIGID.CLIPOPENAI.10': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.10),
    'RIGID.CLIPOPENAI.20': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.20),
    'RIGID.CLIPOPENAI.30': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.30),
    'RIGID.CLIPOPENAI.50': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.50),
    'RIGID.CLIPOPENAI.75': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.75),
    'RIGID.CLIPOPENAI.100': lambda: RIGIDOpenAICLIPHistogram(noise_level=1.0),


    # ===== ConvNeXt =====
    'RIGID.CONVNEXT.001': lambda: RIGIDConvNeXtHistogram(noise_level=0.001),
    'RIGID.CONVNEXT.01': lambda: RIGIDConvNeXtHistogram(noise_level=0.01),
    'RIGID.CONVNEXT.05': lambda: RIGIDConvNeXtHistogram(noise_level=0.05),
    'RIGID.CONVNEXT.10': lambda: RIGIDConvNeXtHistogram(noise_level=0.10),
    'RIGID.CONVNEXT.20': lambda: RIGIDConvNeXtHistogram(noise_level=0.20),
    'RIGID.CONVNEXT.30': lambda: RIGIDConvNeXtHistogram(noise_level=0.30),
    'RIGID.CONVNEXT.50': lambda: RIGIDConvNeXtHistogram(noise_level=0.50),
    'RIGID.CONVNEXT.75': lambda: RIGIDConvNeXtHistogram(noise_level=0.75),
    'RIGID.CONVNEXT.100': lambda: RIGIDConvNeXtHistogram(noise_level=1.0),

    # ===== ResNet-50 =====
    'RIGID.RESNET.001': lambda: RIGIDResNet50Histogram(noise_level=0.001),
    'RIGID.RESNET.01': lambda: RIGIDResNet50Histogram(noise_level=0.01),
    'RIGID.RESNET.05': lambda: RIGIDResNet50Histogram(noise_level=0.05),
    'RIGID.RESNET.10': lambda: RIGIDResNet50Histogram(noise_level=0.10),
    'RIGID.RESNET.20': lambda: RIGIDResNet50Histogram(noise_level=0.20),
    'RIGID.RESNET.30': lambda: RIGIDResNet50Histogram(noise_level=0.30),
    'RIGID.RESNET.50': lambda: RIGIDResNet50Histogram(noise_level=0.50),
    'RIGID.RESNET.75': lambda: RIGIDResNet50Histogram(noise_level=0.75),
    'RIGID.RESNET.100': lambda: RIGIDResNet50Histogram(noise_level=1.0),
}


def get_histogram_generator(statistic: str) -> Optional[object]:
    """Returns the appropriate histogram generator based on statistic name and level."""
    return STATISTIC_HISTOGRAMS.get(statistic)()
