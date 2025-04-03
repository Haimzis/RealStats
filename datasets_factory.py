from enum import Enum

from data_utils import CocoDataset, ImageDataset, ProGanDataset

# DatasetType Enum with dataset paths and corresponding dataset classes
class DatasetType(Enum):
    CELEBA = {
        "train_real": {"path": "data/CelebaHQMaskDataset/train/images_faces", "class": ImageDataset},
        "test_real": {"path": "data/CelebaHQMaskDataset/test/images_faces", "class": ImageDataset},
        "train_fake": {"path": "data/stable-diffusion-face-dataset/1024/both_faces", "class": ImageDataset},
        "test_fake": {"path": "data/stable-diffusion-face-dataset/1024/both_faces", "class": ImageDataset}
    }

    PROGAN_FACES = {
        "train_real": {"path": "data/CNNDetector/testset/whichfaceisreal_extracted/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/whichfaceisreal_extracted/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/testset/whichfaceisreal_extracted/1_fake", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/whichfaceisreal_extracted/1_fake", "class": ImageDataset}
    }

    COCO = {
        "train_real": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "train_fake": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_fake": {"path": "data/CLIPDetector/test_set/fake/sdxl_cocoval", "class": ImageDataset}
    }

    # COCO_ALL = {
    #     "train_real": {"path": "data/CLIPDetector/train_set/coco2017/train2017", "class": ImageDataset},
    #     "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
    #     "train_fake": {"path": "data/CLIPDetector/train_set/coco_latent_t2i/train2017", "class": ImageDataset},
    #     "test_fake": {"path": "data/CLIPDetector/test_set/fake/sdxl_cocoval", "class": ImageDataset}
    # }

    # COCO_LEAKAGE = {
    #     "train_real": {"path": "data/COCO_LEAKAGE/train/real", "class": ImageDataset},
    #     "test_real": {"path": "data/COCO_LEAKAGE/test/real", "class": ImageDataset},
    #     "train_fake": {"path": "data/COCO_LEAKAGE/train/fake", "class": ImageDataset},
    #     "test_fake": {"path": "data/COCO_LEAKAGE/test/fake", "class": ImageDataset}
    # }

    PROGAN_FACES_BUT_CELEBA_AS_TRAIN = {
        "train_real": {"path": "data/CelebaHQMaskDataset/train/images_faces", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/whichfaceisreal_extracted/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/stable-diffusion-face-dataset/1024/both_faces", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/whichfaceisreal_extracted/1_fake", "class": ImageDataset}
    }

    COCO_BIGGAN_256 = {
        "train_real": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "train_fake": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_fake": {"path": "data/CLIPDetector/test_set/fake/biggan_256", "class": ImageDataset}
    }

    COCO_STABLE_DIFFUSION_XL = {
        "train_real": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "train_fake": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_fake": {"path": "data/CLIPDetector/synthbuster/stable-diffusion-xl", "class": ImageDataset}
    }

    COCO_DALLE3_COCOVAL = {
        "train_real": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "train_fake": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_fake": {"path": "data/CLIPDetector/test_set/fake/dalle3_cocoval", "class": ImageDataset}
    }

    COCO_SYNTH_MIDJOURNEY_V5 = {
        "train_real": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "train_fake": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_fake": {"path": "data/CLIPDetector/synthbuster/midjourney-v5", "class": ImageDataset}
    }

    COCO_STABLE_DIFFUSION_2 = {
        "train_real": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "train_fake": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_fake": {"path": "data/CLIPDetector/synthbuster/stable-diffusion-2", "class": ImageDataset}
    }

    COCO_STABLE_DIFFUSION_2_768 = {
        "train_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "train_fake": {"path": "data/CLIPDetector/test_set/fake/stable_diffusion_2_1_768", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/test_set/fake/stable_diffusion_2_1_768", "class": ImageDataset}
    }

    PROGAN = {
        "train_real": {"path": "data/CNNDetector/trainset", "class": ProGanDataset},
        "test_real": {"path": "data/CNNDetector/testset/progan", "class": ProGanDataset},
        "train_fake": {"path": "data/CNNDetector/trainset", "class": ProGanDataset},
        "test_fake": {"path": "data/CNNDetector/testset/progan", "class": ProGanDataset}
    }
    
    PROGAN_BIGGAN = {
        "train_real": {"path": "data/CNNDetector/trainset", "class": ProGanDataset},
        "test_real": {"path": "data/CNNDetector/testset/biggan/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/trainset", "class": ProGanDataset},
        "test_fake": {"path": "data/CNNDetector/testset/biggan/1_fake", "class": ImageDataset}
    }

    PROGAN_LDM = {
        "train_real": {"path": "data/CNNDetector/trainset", "class": ProGanDataset},
        # "test_real": {"path": "data/CNNDetector/testset/biggan/0_real", "class": ProGanDataset},
        "train_fake": {"path": "data/CNNDetector/trainset", "class": ProGanDataset},
        "test_fake": {"path": "data/CNNDetector/diffusion_datasets/ldm_200/1_fake", "class": ImageDataset}
    }

    PROGAN_DALLE = {
        "train_real": {"path": "data/CNNDetector/trainset", "class": ProGanDataset},
        # "test_real": {"path": "data/CNNDetector/testset/biggan/0_real", "class": ProGanDataset},
        "train_fake": {"path": "data/CNNDetector/trainset", "class": ProGanDataset},
        "test_fake": {"path": "data/CNNDetector/diffusion_datasets/dalle/1_fake", "class": ImageDataset}
    }
    
    BIGGAN_TEST_ONLY = {
        "train_real": {"path": "data/CNNDetector/testset/biggan/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/biggan/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/testset/biggan/1_fake", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/biggan/1_fake", "class": ImageDataset}
    }

    CYCLEGAN_TEST_ONLY = {
        "train_real": {"path": "data/CNNDetector/testset/cyclegan/apple/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/cyclegan/apple/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/testset/cyclegan/apple/1_fake", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/cyclegan/apple/1_fake", "class": ImageDataset}
    }

    GAUGAN_TEST_ONLY = {
        "train_real": {"path": "data/CNNDetector/testset/gaugan/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/gaugan/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/testset/gaugan/1_fake", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/gaugan/1_fake", "class": ImageDataset}
    }

    PROGAN_TEST_ONLY = {
        "train_real": {"path": "data/CNNDetector/testset/progan/airplane/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/progan/airplane/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/testset/progan/airplane/1_fake", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/progan/airplane/1_fake", "class": ImageDataset}
    }

    SEEINGDARK_TEST_ONLY = {
        "train_real": {"path": "data/CNNDetector/testset/seeingdark/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/seeingdark/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/testset/seeingdark/1_fake", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/seeingdark/1_fake", "class": ImageDataset}
    }

    STYLEGAN_TEST_ONLY = {
        "train_real": {"path": "data/CNNDetector/testset/stylegan/car/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/stylegan/car/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/testset/stylegan/car/1_fake", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/stylegan/car/1_fake", "class": ImageDataset}
    }

    CRN_TEST_ONLY = {
        "train_real": {"path": "data/CNNDetector/testset/crn/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/crn/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/testset/crn/1_fake", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/crn/1_fake", "class": ImageDataset}
    }

    DEEPFAKE_TEST_ONLY = {
        "train_real": {"path": "data/CNNDetector/testset/deepfake/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/deepfake/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/testset/deepfake/1_fake", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/deepfake/1_fake", "class": ImageDataset}
    }

    IMLE_TEST_ONLY = {
        "train_real": {"path": "data/CNNDetector/testset/imle/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/imle/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/testset/imle/1_fake", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/imle/1_fake", "class": ImageDataset}
    }

    SAN_TEST_ONLY = {
        "train_real": {"path": "data/CNNDetector/testset/san/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/san/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/testset/san/1_fake", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/san/1_fake", "class": ImageDataset}
    }

    STARGAN_TEST_ONLY = {
        "train_real": {"path": "data/CNNDetector/testset/stargan/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/stargan/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/testset/stargan/1_fake", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/stargan/1_fake", "class": ImageDataset}
    }

    STYLEGAN2_TEST_ONLY = {
        "train_real": {"path": "data/CNNDetector/testset/stylegan2/car/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/stylegan2/car/0_real", "class": ImageDataset},
        "train_fake": {"path": "data/CNNDetector/testset/stylegan2/car/1_fake", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/stylegan2/car/1_fake", "class": ImageDataset}
    }

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
            dataset_type (str): The dataset type (e.g., 'CelebA', 'ProGan', 'COCO').
            transform (callable, optional): Transform to apply to the images.

        Returns:
            dict: Dictionary containing dataset instances for train_real, test_real, train_fake, test_fake
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
