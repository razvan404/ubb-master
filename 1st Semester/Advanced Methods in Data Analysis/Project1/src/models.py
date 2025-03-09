import os
from pathlib import Path

from ultralytics import YOLO as BaseYOLO

from .filesystem import FileSystem


def YOLO(model_name: str = "yolov8s.pt", *, models_path: Path = None) -> BaseYOLO:
    if models_path is None:
        models_path = FileSystem.MODELS_DIR

    os.makedirs(models_path, exist_ok=True)
    model_path = models_path / model_name

    return BaseYOLO(model=model_path, task="detect", verbose=False)
