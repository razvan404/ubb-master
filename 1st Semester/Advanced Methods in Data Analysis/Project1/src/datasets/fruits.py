from .traffic_signs import TrafficSignsDataset
from ..filesystem import FileSystem


class FruitsDataset(TrafficSignsDataset):
    directories_mapping = {"train": "train", "val": "valid", "test": "test"}
    default_dataset_path = FileSystem.DATASETS_DIR / "Fruits"
