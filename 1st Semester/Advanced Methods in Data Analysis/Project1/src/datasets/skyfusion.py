import json
import os.path
import shutil
from pathlib import Path

import pandas as pd
import yaml

from .yolo_dataset import YoloDataset
from ..filesystem import FileSystem
from ..transforms import MinPointAndSizeBBox


class SkyFusionDataset(YoloDataset):
    directories_mapping = {"train": "train", "val": "valid", "test": "test"}
    annotation_json = "_annotations.coco.json"
    default_dataset_path = FileSystem.DATASETS_DIR / "SkyFusion"
    default_preprocessed_dir = "preprocessed"
    data_yaml_file = "data.yaml"
    categories_mapping = {
        1: {"internal_id": 0, "name": "aircraft"},
        2: {"internal_id": 1, "name": "ship"},
        3: {"internal_id": 2, "name": "vehicle"},
    }
    _category_names = [
        category["name"]
        for category in sorted(
            categories_mapping.values(), key=lambda category: category["internal_id"]
        )
    ]

    @classmethod
    def export_to_yolo(
        cls, dataset_path: Path | None = None, force: bool = False
    ) -> Path:
        if dataset_path is None:
            dataset_path = cls.default_dataset_path
        destination_dir = dataset_path / cls.default_preprocessed_dir

        if (
            os.path.exists(data_path := destination_dir / cls.data_yaml_file)
            and not force
        ):
            return data_path

        data_yaml = {"path": os.path.relpath(destination_dir, FileSystem.DATASETS_DIR)}

        for key, directory in cls.directories_mapping.items():
            images_dir = cls._convert_coco_to_yolo(
                dataset_path / directory, destination_dir / directory, force=force
            )

            data_yaml[key] = os.path.relpath(images_dir, destination_dir)

        data_yaml["nc"] = len(cls._category_names)
        data_yaml["names"] = cls._category_names
        with open(data_path, "w") as f:
            yaml.dump(data_yaml, f)

        return data_path

    @classmethod
    def _load_annotations_json(cls, json_path: Path):
        with open(json_path) as f:
            raw_json = json.load(f)

        categories = pd.DataFrame(raw_json["categories"])[["id", "name"]]
        images = pd.DataFrame(raw_json["images"])[
            ["id", "file_name", "height", "width"]
        ]
        annotations = pd.DataFrame(raw_json["annotations"])[
            ["id", "image_id", "category_id", "bbox", "area"]
        ]

        return categories, images, annotations

    @classmethod
    def _label_filename(cls, image_filename: str):
        return os.path.splitext(image_filename)[0] + ".txt"

    @classmethod
    def _convert_coco_to_yolo(
        cls, source_dir: Path, destination_dir: Path, force: bool = False
    ):
        coco_json_path = source_dir / cls.annotation_json
        categories, images, annotations = cls._load_annotations_json(coco_json_path)

        images_new_dir = destination_dir / cls.images_dir
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

        for idx, image in images.iterrows():
            image_id = image["id"]

            image_annotations = [
                (
                    cls.categories_mapping[annotation["category_id"]]["internal_id"],
                    *MinPointAndSizeBBox.preprocess(
                        annotation["bbox"], image["width"], image["height"]
                    ),
                )
                for _, annotation in annotations[
                    annotations["image_id"] == image_id
                ].iterrows()
            ]

            image_old_filename = image["file_name"]
            image_new_filename = (
                f"img_{idx:05d}{os.path.splitext(image_old_filename)[1]}"
            )
            shutil.copy(
                source_dir / image_old_filename, images_new_dir / image_new_filename
            )

            label_filename = cls._label_filename(image_new_filename)
            with open(labels_new_dir / label_filename, "w") as label_file:
                label_file.write(
                    "\n".join(
                        [
                            " ".join(map(str, image_annotation))
                            for image_annotation in image_annotations
                        ]
                    )
                )

        return images_new_dir
