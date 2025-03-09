import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patheffects import Stroke, Normal


class ImageVisualizations:
    @classmethod
    def plot_image_with_annotations(
        cls, image: Image, bboxes: torch.Tensor, categories: list[str]
    ):
        plt.imshow(image)
        for bbox, category in zip(bboxes, categories):
            x_min, y_min, x_max, y_max = bbox
            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            plt.gca().add_patch(rect)

            plt.text(
                (x_min + x_max) / 2,
                y_min - 5,
                category,
                color="white",
                fontsize=8,
                ha="center",
                va="bottom",
                path_effects=[
                    Stroke(linewidth=1, foreground="black"),
                    Normal(),
                ],
            )
        plt.axis("off")
