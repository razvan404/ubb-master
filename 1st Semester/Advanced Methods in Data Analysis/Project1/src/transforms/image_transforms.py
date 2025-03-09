import torch
from torchvision import transforms

IMAGES_RESOLUTION = 640


def to_internal_representation_transform(resolution: int):
    class CenterCropDynamic:
        def __call__(self, img: torch.Tensor):
            min_side = min(img.shape[1:3])
            return transforms.CenterCrop(min_side)(img)

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            CenterCropDynamic(),
            transforms.Resize((resolution, resolution), antialias=True),
        ]
    )


def to_external_representation_transform():
    return transforms.Compose(
        [
            transforms.Lambda(lambda x: x * 0.5 + 0.5),
            transforms.ToPILImage(),
        ]
    )


class ImageTransforms:
    preprocess = to_internal_representation_transform(IMAGES_RESOLUTION)
    postprocess = to_external_representation_transform()
