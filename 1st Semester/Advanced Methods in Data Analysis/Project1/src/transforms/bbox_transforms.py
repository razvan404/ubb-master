from abc import abstractmethod

from .image_transforms import IMAGES_RESOLUTION


class YoloBBox:
    @classmethod
    @abstractmethod
    def preprocess(
        cls,
        bbox: tuple,
        image_width: int,
        image_height: int,
    ): ...

    @classmethod
    def postprocess(
        cls,
        bbox: tuple,
        image_width: int = IMAGES_RESOLUTION,
        image_height: int = IMAGES_RESOLUTION,
    ):
        x_center, y_center, box_width, box_height = bbox
        x_min = (x_center - box_width / 2) * image_width
        y_min = (y_center - box_height / 2) * image_height
        x_max = (x_center + box_width / 2) * image_width
        y_max = (y_center + box_height / 2) * image_height

        return x_min, y_min, x_max, y_max


class MinPointAndSizeBBox(YoloBBox):
    @classmethod
    def preprocess(
        cls,
        bbox: tuple,
        image_width: int,
        image_height: int,
    ):
        x_min, y_min, box_width, box_height = bbox
        x_center = x_min + box_width / 2
        y_center = y_min + box_height / 2

        x_center /= image_width
        y_center /= image_height
        box_width /= image_width
        box_height /= image_height

        return x_center, y_center, box_width, box_height
