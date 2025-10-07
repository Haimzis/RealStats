from enum import Enum
from data_utils import CocoDataset, ImageDataset, ManifoldBiasDataset, ProGanDataset


class ManifoldBiasGenerator(Enum):
    CNNSPOT_BIGGAN = " CNNSpot_test/biggan"
    CNNSPOT_CRN = " CNNSpot_test/crn"
    CNNSPOT_CYCLEGAN = " CNNSpot_test/cyclegan"
    CNNSPOT_DEEPFAKE = " CNNSpot_test/deepfake"
    CNNSPOT_GAUGAN = " CNNSpot_test/gaugan"
    CNNSPOT_IMLE = " CNNSpot_test/imle"
    CNNSPOT_PROGAN = "CNNSpot/progan"
    CNNSPOT_SAN = " CNNSpot_test/san"
    CNNSPOT_STARGAN = " CNNSpot_test/stargan"
    CNNSPOT_STYLEGAN2 = " CNNSpot_test/stylegan2"
    CNNSPOT_WHICHFACEISREAL = " CNNSpot_test/whichfaceisreal"
    GENIMAGE_ADM_GENIMAGE = "GenImage/adm_genimage"
    GENIMAGE_MIDJOURNEY_GENIMAGE = "GenImage/midjourney_genimage"
    GENIMAGE_SD_V4_GENIMAGE = "GenImage/sd_v4_genimage"
    GENIMAGE_SD_V5_GENIMAGE = "GenImage/sd_v5_genimage"
    GENIMAGE_VQDM_GENIMAGE = "GenImage/vqdm_genimage"
    GENIMAGE_WUKONG_GENIMAGE = "GenImage/wukong_genimage"
    STABLE_DIFFUSION_FACES_SD2 = "stable_diffusion_human_models/768_sdv2"
    STABLE_DIFFUSION_FACES_SDXL = "stable_diffusion_human_models/1024_sdxl"
    SYNTHBUSTER_DALLE3 = "SynthBuster/dalle3"
    SYNTHBUSTER_MIDJOURNEY_V5 = "SynthBuster/midjourney-v5"
    SYNTHBUSTER_STABLE_DIFFUSION_2 = "SynthBuster/stable-diffusion-2"
    SYNTHBUSTER_STABLE_DIFFUSION_XL = "SynthBuster/stable-diffusion-xl"
    UNIVERSAL_FAKE_DETECT_DALLE = "Universal_Fake_Detect/diffusion_datasets/dalle"
    UNIVERSAL_FAKE_DETECT_GLIDE_100_27 = "Universal_Fake_Detect/diffusion_datasets/glide_100_27"
    UNIVERSAL_FAKE_DETECT_GLIDE_50_27 = "Universal_Fake_Detect/diffusion_datasets/glide_50_27"


def _manifold_bias_entry(generator=None):
    return {
        "reference_real": {
            "path": "data/ManifoldBiasDataset",
            "class": lambda root, label, transform=None: ManifoldBiasDataset(
                root, "reference_real_paths.csv", label, transform
            ),
        },
        "test_real": {
            "path": "data/ManifoldBiasDataset",
            "class": lambda root, label, transform=None: ManifoldBiasDataset(
                root, "test_real_paths.csv", label, transform
            ),
        },
        "test_fake": {
            "path": "data/ManifoldBiasDataset",
            "class": lambda root, label, transform=None: ManifoldBiasDataset(
                root, "test_fake_paths.csv", label, transform, generator=generator
            ),
        },
    }

class DatasetType(Enum):
    # === Base dataset ===
    ALL = _manifold_bias_entry()

    # ===  CNNSpot_test ===
    CNNSPOT_BIGGAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOT_BIGGAN.value)
    CNNSPOT_CRN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOT_CRN.value)
    CNNSPOT_CYCLEGAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOT_CYCLEGAN.value)
    CNNSPOT_DEEPFAKE = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOT_DEEPFAKE.value)
    CNNSPOT_GAUGAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOT_GAUGAN.value)
    CNNSPOT_IMLE = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOT_IMLE.value)
    CNNSPOT_PROGAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOT_PROGAN.value) 
    CNNSPOT_SAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOT_SAN.value)
    CNNSPOT_STARGAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOT_STARGAN.value) 
    CNNSPOT_STYLEGAN2 = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOT_STYLEGAN2.value)
    CNNSPOT_WHICHFACEISREAL = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOT_WHICHFACEISREAL.value) 

    # === GenImage ===
    GENIMAGE_ADM_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_ADM_GENIMAGE.value)
    GENIMAGE_MIDJOURNEY_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_MIDJOURNEY_GENIMAGE.value)
    GENIMAGE_SD_V4_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_SD_V4_GENIMAGE.value)
    GENIMAGE_SD_V5_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_SD_V5_GENIMAGE.value)
    GENIMAGE_VQDM_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_VQDM_GENIMAGE.value)
    GENIMAGE_WUKONG_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_WUKONG_GENIMAGE.value)

    # === Stable Diffusion Faces Models ===
    STABLE_DIFFUSION_FACES_SD2 = _manifold_bias_entry(ManifoldBiasGenerator.STABLE_DIFFUSION_FACES_SD2.value)
    STABLE_DIFFUSION_FACES_SDXL = _manifold_bias_entry(ManifoldBiasGenerator.STABLE_DIFFUSION_FACES_SDXL.value)

    # === SynthBuster ===
    SYNTHBUSTER_DALLE3 = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_DALLE3.value)
    SYNTHBUSTER_MIDJOURNEY_V5 = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_MIDJOURNEY_V5.value)
    SYNTHBUSTER_STABLE_DIFFUSION_2 = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_STABLE_DIFFUSION_2.value)
    SYNTHBUSTER_STABLE_DIFFUSION_XL = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_STABLE_DIFFUSION_XL.value)

    # === Universal Fake Detect ===
    UNIVERSAL_FAKE_DETECT_DALLE = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_DALLE.value)
    UNIVERSAL_FAKE_DETECT_GLIDE_100_27 = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GLIDE_100_27.value)
    UNIVERSAL_FAKE_DETECT_GLIDE_50_27 = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GLIDE_50_27.value)

    def get_paths(self):
        return self.value


# Dataset Factory
class DatasetFactory:
    """Factory class for creating datasets based on dataset type."""

    @staticmethod
    def create_dataset(dataset_type, transform=None):
        """
        Create datasets dynamically based on dataset type.

        Args:
            dataset_type (str): The dataset type (e.g., 'ALL').
            transform (callable, optional): Transform to apply to the images.

        Returns:
            dict: Dictionary containing dataset instances for reference_real, test_real, test_fake
        """
        dataset_type = dataset_type.upper()

        # Ensure dataset_type exists in DatasetType
        if dataset_type not in DatasetType.__members__:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        # Retrieve dataset configuration
        dataset_info = DatasetType[dataset_type].get_paths()

        datasets = {}
        for split, split_info in dataset_info.items():
            dataset_class = split_info["class"]
            dataset_path = split_info["path"]

            datasets[split] = dataset_class(dataset_path, 0 if "real" in split else 1, transform=transform)

        return datasets
