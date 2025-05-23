from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cyclopts import App

app = App()


def generate_palette(num_classes: int) -> list[tuple[int, int, int]]:
    """Generate a color palette with `num_classes` distinct colors.

    Colors are generated using matplotlib colormap and converted to BGR.
    """
    cmap = plt.get_cmap("hsv", num_classes)
    palette = []
    for i in range(num_classes):
        rgb = np.array(cmap(i)[:3]) * 255  # RGB values
        bgr = tuple(int(c) for c in rgb[::-1])  # Convert to BGR for OpenCV
        palette.append(bgr)
    return palette


def apply_colormap(
    mask: np.ndarray, palette: list[tuple[int, int, int]]
) -> np.ndarray:
    """Apply a color palette to a segmentation mask.

    mask: 2D array of shape (h, w) with class indices.
    palette: list of BGR color tuples.
    """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(palette):
        color_mask[mask == class_idx] = color
    return color_mask


def visualize_npy_file(file_path: Path) -> None:
    data = np.load(file_path)

    if data.ndim != 4:
        raise ValueError(
            f"Expected shape (batch_size, h, w, n_classes), got {data.shape}"
        )

    batch_size, n_classes, _, _ = data.shape
    palette = generate_palette(n_classes)

    for i in range(batch_size):
        mask = np.argmax(data[i], axis=0).astype(np.uint8)
        colored_mask = apply_colormap(mask, palette)
        cv2.imshow(f"{file_path.name} - Sample {i}", colored_mask)
        key = cv2.waitKey(0)
        if key in {27, ord("q")}:
            return


@app.default
def main(path: Path) -> None:
    try:
        visualize_npy_file(path)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
