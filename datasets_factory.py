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




def _manifold_bias_entry(generator=None, group_leakage=False):
    suffix = "_group_leakage" if group_leakage else ""
    return {
        "reference_real": {
            "path": "data/ManifoldBiasDataset",
            "class": lambda root, label, transform=None: ManifoldBiasDataset(
                root, f"reference_real_paths{suffix}_30.csv", label, transform
            ),
        },
        "test_real": {
            "path": "data/ManifoldBiasDataset",
            "class": lambda root, label, transform=None: ManifoldBiasDataset(
                root, f"test_real_paths{suffix}_30.csv", label, transform
            ),
        },
        "test_fake": {
            "path": "data/ManifoldBiasDataset",
            "class": lambda root, label, transform=None: ManifoldBiasDataset(
                root, "test_fake_paths_extended.csv", label, transform, generator=generator
            ),
        },
    }

# DatasetType Enum with dataset paths and corresponding dataset classes
class DatasetType_AAAI(Enum):
    CELEBA = {
        "reference_real": {"path": "data/CelebaHQMaskDataset/train/images_faces", "class": ImageDataset},
        "test_real": {"path": "data/CelebaHQMaskDataset/test/images_faces", "class": ImageDataset},
        "test_fake": {"path": "data/stable-diffusion-face-dataset/1024/both_faces", "class": ImageDataset}
    }

    PROGAN_FACES = {
        "reference_real": {"path": "data/CNNDetector/testset/whichfaceisreal_extracted/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/whichfaceisreal_extracted/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/whichfaceisreal_extracted/1_fake", "class": ImageDataset}
    }

    COCO = {
        "reference_real": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/test_set/fake/sdxl_cocoval", "class": ImageDataset}
    }

    # COCO_ALL = {
    #     "reference_real": {"path": "data/CLIPDetector/train_set/coco2017/train2017", "class": ImageDataset},
    #     "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
    #     "test_fake": {"path": "data/CLIPDetector/test_set/fake/sdxl_cocoval", "class": ImageDataset}
    # }

    # COCO_LEAKAGE = {
    #     "reference_real": {"path": "data/COCO_LEAKAGE/train/real", "class": ImageDataset},
    #     "test_real": {"path": "data/COCO_LEAKAGE/test/real", "class": ImageDataset},
    #     "test_fake": {"path": "data/COCO_LEAKAGE/test/fake", "class": ImageDataset}
    # }

    PROGAN_FACES_BUT_CELEBA_AS_TRAIN = {
        "reference_real": {"path": "data/CelebaHQMaskDataset/train/images_faces", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/whichfaceisreal_extracted/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/whichfaceisreal_extracted/1_fake", "class": ImageDataset}
    }

    COCO_BIGGAN_256 = {
        "reference_real": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/test_set/fake/biggan_256", "class": ImageDataset}
    }

    COCO_STABLE_DIFFUSION_XL = {
        "reference_real": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/synthbuster/stable-diffusion-xl", "class": ImageDataset}
    }

    COCO_DALLE3_COCOVAL = {
        "reference_real": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/test_set/fake/dalle3_cocoval", "class": ImageDataset}
    }

    COCO_SYNTH_MIDJOURNEY_V5 = {
        "reference_real": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/synthbuster/midjourney-v5", "class": ImageDataset}
    }

    COCO_STABLE_DIFFUSION_2 = {
        "reference_real": {"path": "data/CLIPDetector/train_set/", "class": CocoDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/synthbuster/stable-diffusion-2", "class": ImageDataset}
    }

    COCO_STABLE_DIFFUSION_2_768 = {
        "reference_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/test_set/fake/stable_diffusion_2_1_768", "class": ImageDataset}
    }

    PROGAN = {
        "reference_real": {"path": "data/CNNDetector/trainset", "class": ProGanDataset},
        "test_real": {"path": "data/CNNDetector/testset/progan", "class": ProGanDataset},
        "test_fake": {"path": "data/CNNDetector/testset/progan", "class": ProGanDataset}
    }
    
    PROGAN_BIGGAN = {
        "reference_real": {"path": "data/CNNDetector/trainset", "class": ProGanDataset},
        "test_real": {"path": "data/CNNDetector/testset/biggan/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/biggan/1_fake", "class": ImageDataset}
    }

    PROGAN_DALLE = {
        "reference_real": {"path": "data/CNNDetector/trainset", "class": ProGanDataset},
        "test_real": {"path": "data/CNNDetector/testset/biggan/0_real", "class": ProGanDataset},
        "test_fake": {"path": "data/CNNDetector/diffusion_datasets/dalle/1_fake", "class": ImageDataset}
    }
    
    BIGGAN_TEST_ONLY = {
        "reference_real": {"path": "data/CNNDetector/testset/biggan/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/biggan/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/biggan/1_fake", "class": ImageDataset}
    }

    CYCLEGAN_TEST_ONLY = {
        "reference_real": {"path": "data/CNNDetector/testset/cyclegan/apple/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/cyclegan/apple/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/cyclegan/apple/1_fake", "class": ImageDataset}
    }

    GAUGAN_TEST_ONLY = {
        "reference_real": {"path": "data/CNNDetector/testset/gaugan/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/gaugan/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/gaugan/1_fake", "class": ImageDataset}
    }

    PROGAN_TEST_ONLY = {
        "reference_real": {"path": "data/CNNDetector/testset/progan/airplane/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/progan/airplane/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/progan/airplane/1_fake", "class": ImageDataset}
    }

    SEEINGDARK_TEST_ONLY = {
        "reference_real": {"path": "data/CNNDetector/testset/seeingdark/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/seeingdark/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/seeingdark/1_fake", "class": ImageDataset}
    }

    STYLEGAN_TEST_ONLY = {
        "reference_real": {"path": "data/CNNDetector/testset/stylegan/car/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/stylegan/car/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/stylegan/car/1_fake", "class": ImageDataset}
    }

    CRN_TEST_ONLY = {
        "reference_real": {"path": "data/CNNDetector/testset/crn/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/crn/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/crn/1_fake", "class": ImageDataset}
    }

    DEEPFAKE_TEST_ONLY = {
        "reference_real": {"path": "data/CNNDetector/testset/deepfake/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/deepfake/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/deepfake/1_fake", "class": ImageDataset}
    }

    IMLE_TEST_ONLY = {
        "reference_real": {"path": "data/CNNDetector/testset/imle/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/imle/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/imle/1_fake", "class": ImageDataset}
    }

    SAN_TEST_ONLY = {
        "reference_real": {"path": "data/CNNDetector/testset/san/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/san/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/san/1_fake", "class": ImageDataset}
    }

    STARGAN_TEST_ONLY = {
        "reference_real": {"path": "data/CNNDetector/testset/stargan/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/stargan/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/stargan/1_fake", "class": ImageDataset}
    }

    STYLEGAN2_TEST_ONLY = {
        "reference_real": {"path": "data/CNNDetector/testset/stylegan2/car/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/testset/stylegan2/car/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/testset/stylegan2/car/1_fake", "class": ImageDataset}
    }

    # TEST ONLY
    CELEBA_TEST_ONLY = {
        "reference_real": {"path": "data/CelebaHQMaskDataset/test/images_faces", "class": ImageDataset},
        "test_real": {"path": "data/CelebaHQMaskDataset/test/images_faces", "class": ImageDataset},
        "test_fake": {"path": "data/stable-diffusion-face-dataset/1024/both_faces", "class": ImageDataset}
    }

    COCO_TEST_ONLY = {
        "reference_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/test_set/fake/sdxl_cocoval", "class": ImageDataset}
    }

    COCO_BIGGAN_256_TEST_ONLY = {
        "reference_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/test_set/fake/biggan_256", "class": ImageDataset}
    }

    COCO_STABLE_DIFFUSION_XL_TEST_ONLY = {
        "reference_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/synthbuster/stable-diffusion-xl", "class": ImageDataset}
    }

    COCO_DALLE3_COCOVAL_TEST_ONLY = {
        "reference_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/test_set/fake/dalle3_cocoval", "class": ImageDataset}
    }

    COCO_SYNTH_MIDJOURNEY_V5_TEST_ONLY = {
        "reference_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/synthbuster/midjourney-v5", "class": ImageDataset}
    }

    COCO_STABLE_DIFFUSION_2_TEST_ONLY = {
        "reference_real":  {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/synthbuster/stable-diffusion-2", "class": ImageDataset}
    }

    # COMPLETION DATASETS
    COCO_STABLE_DIFFUSION_1_4_TEST_ONLY = {
        "reference_real":  {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_real": {"path": "data/CLIPDetector/test_set/real/real_coco_valid", "class": ImageDataset},
        "test_fake": {"path": "data/CLIPDetector/synthbuster/stable-diffusion-1-4", "class": ImageDataset}
    }

    IMAGENET_GLIDE_100_10 = {
        "reference_real": {"path": "data/CNNDetector/diffusion_datasets/imagenet/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/diffusion_datasets/imagenet/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/diffusion_datasets/glide_100_10/1_fake", "class": ImageDataset}
    }

    IMAGENET_GLIDE_100_27 = {
        "reference_real": {"path": "data/CNNDetector/diffusion_datasets/imagenet/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/diffusion_datasets/imagenet/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/diffusion_datasets/glide_100_27/1_fake", "class": ImageDataset}
    }

    IMAGENET_LDM = {
        "reference_real": {"path": "data/CNNDetector/diffusion_datasets/imagenet/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/diffusion_datasets/imagenet/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/diffusion_datasets/ldm_200_cfg/1_fake", "class": ImageDataset}
    }

    LAION_GLIDE_100_10 = {
        "reference_real": {"path": "data/CNNDetector/diffusion_datasets/laion/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/diffusion_datasets/laion/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/diffusion_datasets/glide_100_10/1_fake", "class": ImageDataset}
    }

    LAION_GLIDE_100_27 = {
        "reference_real": {"path": "data/CNNDetector/diffusion_datasets/laion/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/diffusion_datasets/laion/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/diffusion_datasets/glide_100_27/1_fake", "class": ImageDataset}
    }

    LAION_LDM = {
        "reference_real": {"path": "data/CNNDetector/diffusion_datasets/laion/0_real", "class": ImageDataset},
        "test_real": {"path": "data/CNNDetector/diffusion_datasets/laion/0_real", "class": ImageDataset},
        "test_fake": {"path": "data/CNNDetector/diffusion_datasets/ldm_200_cfg/1_fake", "class": ImageDataset}
    }

    def get_paths(self):
        return self.value


# DatasetType Enum with dataset paths and corresponding dataset classes
class DatasetType(Enum):
    # === Base dataset ===
    ALL = _manifold_bias_entry()
    ALL_GROUP_LEAKAGE = _manifold_bias_entry(group_leakage=True)

    # === CNNSpotset ===
    CNNSPOTSET_BIGGAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_BIGGAN.value)
    CNNSPOTSET_BIGGAN_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_BIGGAN.value, group_leakage=True)

    CNNSPOTSET_CRN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_CRN.value)
    CNNSPOTSET_CRN_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_CRN.value, group_leakage=True)

    CNNSPOTSET_CYCLEGAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_CYCLEGAN.value)
    CNNSPOTSET_CYCLEGAN_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_CYCLEGAN.value, group_leakage=True)

    CNNSPOTSET_GAUGAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_GAUGAN.value)
    CNNSPOTSET_GAUGAN_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_GAUGAN.value, group_leakage=True)

    CNNSPOTSET_IMLE = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_IMLE.value)
    CNNSPOTSET_IMLE_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_IMLE.value, group_leakage=True)

    CNNSPOTSET_SAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_SAN.value)
    CNNSPOTSET_SAN_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_SAN.value, group_leakage=True)

    CNNSPOTSET_STYLEGAN = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_STYLEGAN.value)
    CNNSPOTSET_STYLEGAN_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_STYLEGAN.value, group_leakage=True)

    CNNSPOTSET_STYLEGAN2 = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_STYLEGAN2.value)
    CNNSPOTSET_STYLEGAN2_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.CNNSPOTSET_STYLEGAN2.value, group_leakage=True)

    # === GenImage ===
    GENIMAGE_ADM_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_ADM_GENIMAGE.value)
    GENIMAGE_ADM_GENIMAGE_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_ADM_GENIMAGE.value, group_leakage=True)

    GENIMAGE_MIDJOURNEY_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_MIDJOURNEY_GENIMAGE.value)
    GENIMAGE_MIDJOURNEY_GENIMAGE_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_MIDJOURNEY_GENIMAGE.value, group_leakage=True)

    GENIMAGE_SD_V4_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_SD_V4_GENIMAGE.value)
    GENIMAGE_SD_V4_GENIMAGE_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_SD_V4_GENIMAGE.value, group_leakage=True)

    GENIMAGE_SD_V5_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_SD_V5_GENIMAGE.value)
    GENIMAGE_SD_V5_GENIMAGE_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_SD_V5_GENIMAGE.value, group_leakage=True)

    GENIMAGE_VDQM_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_VDQM_GENIMAGE.value)
    GENIMAGE_VDQM_GENIMAGE_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_VDQM_GENIMAGE.value, group_leakage=True)

    GENIMAGE_WUKONG_GENIMAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_WUKONG_GENIMAGE.value)
    GENIMAGE_WUKONG_GENIMAGE_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.GENIMAGE_WUKONG_GENIMAGE.value, group_leakage=True)

    # === Universal Fake Detect ===
    UNIVERSAL_FAKE_DETECT_DALLE = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_DALLE.value)
    UNIVERSAL_FAKE_DETECT_DALLE_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_DALLE.value, group_leakage=True)

    UNIVERSAL_FAKE_DETECT_GLIDE_100_10 = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GLIDE_100_10.value)
    UNIVERSAL_FAKE_DETECT_GLIDE_100_10_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GLIDE_100_10.value, group_leakage=True)

    UNIVERSAL_FAKE_DETECT_GLIDE_100_27 = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GLIDE_100_27.value)
    UNIVERSAL_FAKE_DETECT_GLIDE_100_27_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GLIDE_100_27.value, group_leakage=True)

    UNIVERSAL_FAKE_DETECT_GLIDE_50_27 = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GLIDE_50_27.value)
    UNIVERSAL_FAKE_DETECT_GLIDE_50_27_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GLIDE_50_27.value, group_leakage=True)

    UNIVERSAL_FAKE_DETECT_GUIDED = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GUIDED.value)
    UNIVERSAL_FAKE_DETECT_GUIDED_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_GUIDED.value, group_leakage=True)

    UNIVERSAL_FAKE_DETECT_LDM_100 = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_LDM_100.value)
    UNIVERSAL_FAKE_DETECT_LDM_100_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_LDM_100.value, group_leakage=True)

    UNIVERSAL_FAKE_DETECT_LDM_200 = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_LDM_200.value)
    UNIVERSAL_FAKE_DETECT_LDM_200_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.UNIVERSAL_FAKE_DETECT_LDM_200.value, group_leakage=True)

    SYNTHBUSTER_MIDJOURNEY_V5 = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_MIDJOURNEY_V5.value)
    SYNTHBUSTER_MIDJOURNEY_V5_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_MIDJOURNEY_V5.value, group_leakage=True)

    SYNTHBUSTER_STABLE_DIFFUSION_2 = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_STABLE_DIFFUSION_2.value)
    SYNTHBUSTER_STABLE_DIFFUSION_2_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_STABLE_DIFFUSION_2.value, group_leakage=True)

    SYNTHBUSTER_STABLE_DIFFUSION_XL = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_STABLE_DIFFUSION_XL.value)
    SYNTHBUSTER_STABLE_DIFFUSION_XL_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_STABLE_DIFFUSION_XL.value, group_leakage=True)

    SYNTHBUSTER_DALLE3 = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_DALLE3.value)
    SYNTHBUSTER_DALLE3_GROUP_LEAKAGE = _manifold_bias_entry(ManifoldBiasGenerator.SYNTHBUSTER_DALLE3.value, group_leakage=True)

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
