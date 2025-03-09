import os
import shutil
from pathlib import Path

import yaml

from .yolo_dataset import YoloDataset
from ..filesystem import FileSystem


class TrafficSignsDataset(YoloDataset):
    directories_mapping = {"train": "train", "val": "valid", "test": "test"}
    default_dataset_path = FileSystem.DATASETS_DIR / "TrafficSigns"

    @classmethod
    def export_to_yolo(cls, dataset_path: Path | None = None, force: bool = False):
        if dataset_path is None:
            dataset_path = cls.default_dataset_path
        destination_dir = dataset_path / cls.default_preprocessed_dir

        if (
            os.path.exists(data_path := destination_dir / cls.data_yaml_file)
            and not force
        ):
            return data_path

        new_data_yaml = {
            "path": os.path.relpath(destination_dir, FileSystem.DATASETS_DIR)
        }

        for key, directory in cls.directories_mapping.items():
            images_dir = cls._move_images_and_labels(
                dataset_path / directory, destination_dir / directory, force=force
            )
            new_data_yaml[key] = os.path.relpath(images_dir, destination_dir)

        with open(dataset_path / cls.data_yaml_file, "r") as f:
            old_data_yaml = yaml.safe_load(f)
        new_data_yaml["nc"] = old_data_yaml["nc"]
        new_data_yaml["names"] = old_data_yaml["names"]

        with open(data_path, "w") as f:
            yaml.dump(new_data_yaml, f)

        return data_path

    @classmethod
    def _move_images_and_labels(
        cls, source_dir: Path, destination_dir: Path, force: bool = False
    ):
        images_old_dir = source_dir / cls.images_dir
        images_new_dir = destination_dir / cls.images_dir

        labels_old_dir = source_dir / cls.labels_dir
        labels_new_dir = destination_dir / cls.labels_dir

        if (
            os.path.exists(images_new_dir)
            and os.path.exists(labels_new_dir)
            and not force
        ):
            return images_new_dir

        for directory in [images_new_dir, labels_new_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

        for idx, image_old_filename in enumerate(os.listdir(images_old_dir)):
            label_old_filename = cls._label_filename(image_old_filename)
            image_new_filename = (
                f"img_{idx:05d}{os.path.splitext(image_old_filename)[1]}"
            )
            label_new_filename = cls._label_filename(image_new_filename)

            shutil.copy(
                images_old_dir / image_old_filename, images_new_dir / image_new_filename
            )
            shutil.copy(
                labels_old_dir / label_old_filename, labels_new_dir / label_new_filename
            )

        return images_new_dir
