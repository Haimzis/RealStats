from enum import Enum

from data_utils import CocoDataset, ImageDataset, ManifoldBiasDataset, ProGanDataset


class ManifoldBiasGenerator(Enum):
    CNNSPOTSET_BIGGAN = "CNNSpotset/biggan"
    CNNSPOTSET_CRN = "CNNSpotset/crn"
    CNNSPOTSET_CYCLEGAN = "CNNSpotset/cyclegan"
    CNNSPOTSET_GAUGAN = "CNNSpotset/gaugan"
    CNNSPOTSET_IMLE = "CNNSpotset/imle"
    CNNSPOTSET_SAN = "CNNSpotset/san"
    CNNSPOTSET_STYLEGAN = "CNNSpotset/stylegan"
    CNNSPOTSET_STYLEGAN2 = "CNNSpotset/stylegan2"
    GENIMAGE_ADM_GENIMAGE = "GenImage/adm_genimage"
    GENIMAGE_MIDJOURNEY_GENIMAGE = "GenImage/midjourney_genimage"
    GENIMAGE_SD_V4_GENIMAGE = "GenImage/sd_v4_genimage"
    GENIMAGE_SD_V5_GENIMAGE = "GenImage/sd_v5_genimage"
    GENIMAGE_VDQM_GENIMAGE = "GenImage/vdqm_genimage"
    GENIMAGE_WUKONG_GENIMAGE = "GenImage/wukong_genimage"
    UNIVERSAL_FAKE_DETECT_DALLE = "Universal_Fake_Detect/diffusion_datasets/dalle"
    UNIVERSAL_FAKE_DETECT_GLIDE_100_10 = "Universal_Fake_Detect/diffusion_datasets/glide_100_10"
    UNIVERSAL_FAKE_DETECT_GLIDE_100_27 = "Universal_Fake_Detect/diffusion_datasets/glide_100_27"
    UNIVERSAL_FAKE_DETECT_GLIDE_50_27 = "Universal_Fake_Detect/diffusion_datasets/glide_50_27"
    UNIVERSAL_FAKE_DETECT_GUIDED = "Universal_Fake_Detect/diffusion_datasets/guided"
    UNIVERSAL_FAKE_DETECT_LDM_100 = "Universal_Fake_Detect/diffusion_datasets/ldm_100"
    UNIVERSAL_FAKE_DETECT_LDM_200 = "Universal_Fake_Detect/diffusion_datasets/ldm_200"
    SYNTHBUSTER_MIDJOURNEY_V5 = "SynthBuster/midjourney-v5"
    SYNTHBUSTER_STABLE_DIFFUSION_2 = "SynthBuster/stable-diffusion-2"
    SYNTHBUSTER_STABLE_DIFFUSION_XL = "SynthBuster/stable-diffusion-xl"
    SYNTHBUSTER_DALLE3 = "SynthBuster/dalle3"




def _manifold_bias_entry(generator=None):
    return {
        "reference_real": {
            "path": "data/ManifoldBiasDataset",
            "class": lambda root, label, transform=None: ManifoldBiasDataset(
                root, "reference_real_paths_30.csv", label, transform
            ),
        },
        "test_real": {
            "path": "data/ManifoldBiasDataset",
            "class": lambda root, label, transform=None: ManifoldBiasDataset(
                root, "test_real_paths_30.csv", label, transform
            ),
        },
        "test_fake": {
            "path": "data/ManifoldBiasDataset",
            "class": lambda root, label, transform=None: ManifoldBiasDataset(
                root, "test_fake_paths_extended_old_clean.csv", label, transform, generator=generator
            ),
        },
    }

# DatasetType Enum with dataset paths and corresponding dataset classes
class DatasetType(Enum):
    # === Base dataset ===
    ALL = _manifold_bias_entry()

    # === CNNSpotset ===
    CNNSPOTSET_BIGGAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_BIGGAN.value)

    CNNSPOTSET_CRN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_CRN.value)

    CNNSPOTSET_CYCLEGAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_CYCLEGAN.value)

    CNNSPOTSET_GAUGAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_GAUGAN.value)

    CNNSPOTSET_IMLE = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_IMLE.value)

    CNNSPOTSET_SAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_SAN.value)

    CNNSPOTSET_STYLEGAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_STYLEGAN.value)

    CNNSPOTSET_STYLEGAN2 = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_STYLEGAN2.value)

    # === GenImage ===
    GENIMAGE_ADM_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_ADM_GENIMAGE.value)

    GENIMAGE_MIDJOURNEY_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_MIDJOURNEY_GENIMAGE.value)

    GENIMAGE_SD_V4_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_SD_V4_GENIMAGE.value)

    GENIMAGE_SD_V5_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_SD_V5_GENIMAGE.value)

    GENIMAGE_VDQM_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_VDQM_GENIMAGE.value)

    GENIMAGE_WUKONG_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_WUKONG_GENIMAGE.value)

    # === Universal Fake Detect ===
    UNIVERSAL_FAKE_DETECT_DALLE = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_DALLE.value)

    UNIVERSAL_FAKE_DETECT_GLIDE_100_10 = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GLIDE_100_10.value)

    UNIVERSAL_FAKE_DETECT_GLIDE_100_27 = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GLIDE_100_27.value)

    UNIVERSAL_FAKE_DETECT_GLIDE_50_27 = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GLIDE_50_27.value)

    # UNIVERSAL_FAKE_DETECT_GUIDED = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GUIDED.value)

    # UNIVERSAL_FAKE_DETECT_LDM_100 = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_LDM_100.value)

    # UNIVERSAL_FAKE_DETECT_LDM_200 = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_LDM_200.value)

    SYNTHBUSTER_MIDJOURNEY_V5 = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_MIDJOURNEY_V5.value)

    SYNTHBUSTER_STABLE_DIFFUSION_2 = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_STABLE_DIFFUSION_2.value)

    SYNTHBUSTER_STABLE_DIFFUSION_XL = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_STABLE_DIFFUSION_XL.value)

    SYNTHBUSTER_DALLE3 = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_DALLE3.value)

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
            dataset_type (str): The dataset type (e.g., 'ALL', 'CNNSPOTSET_BIGGAN').
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
