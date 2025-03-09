import os
from abc import ABC, abstractmethod
from pathlib import Path

import yaml
import torch

from PIL import Image
from torch.utils.data import Dataset

from ..transforms.image_transforms import ImageTransforms


class YoloDataset(Dataset, ABC):
    __slots__ = ["default_dataset_path", "directories_mapping"]

    default_preprocessed_dir = "preprocessed"
    data_yaml_file = "data.yaml"
    images_dir = "images"
    labels_dir = "labels"

    def __init__(
        self,
        dataset_path: Path | None = None,
        split: str | None = "train",
    ):
        if dataset_path is None:
            dataset_path = self.default_dataset_path

        if not os.path.exists(
            processed_dataset_path := dataset_path / self.default_preprocessed_dir
        ):
            data_yaml_path = self.export_to_yolo(dataset_path)
        else:
            data_yaml_path = processed_dataset_path / self.data_yaml_file

        with open(data_yaml_path, "r") as f:
            data_yaml = yaml.safe_load(f)

        if split not in self.directories_mapping.keys():
            raise ValueError(f"Invalid split: {split}")

        self.category_names = data_yaml["names"]

        images_path = processed_dataset_path / data_yaml[split]
        labels_path = images_path.parent / self.labels_dir

        self.images_labels_paths = [
            (
                images_path / image_filename,
                labels_path / self._label_filename(image_filename),
            )
            for image_filename in os.listdir(images_path)
        ]

    def __len__(self):
        return len(self.images_labels_paths)

    def __getitem__(self, idx: int):
        image_path, label_path = self.images_labels_paths[idx]
        image = Image.open(image_path)
        image = ImageTransforms.preprocess(image)

        image_annotations = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                annotation = list(map(float, line.strip().split(" ")))
                image_annotations.append(torch.asarray(annotation, dtype=torch.float32))
        image_annotations = torch.stack(image_annotations, dim=0)

        return image, image_annotations

    @classmethod
    @abstractmethod
    def export_to_yolo(cls, dataset_path: Path | None = None, force: bool = False): ...

    @classmethod
    def _label_filename(cls, image_filename: str):
        return os.path.splitext(image_filename)[0] + ".txt"
