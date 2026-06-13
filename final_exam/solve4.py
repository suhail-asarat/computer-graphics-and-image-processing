"""Problem 4: Subtract one image from another."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from textwrap import wrap

import cv2
import matplotlib
import numpy as np

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent


class Tee(StringIO):
    def write(self, text: str) -> int:
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        return super().write(text)


def run_with_gui(main, title: str) -> None:
    capture = Tee()
    with redirect_stdout(capture):
        main()
    lines = [
        part
        for line in capture.getvalue().strip().splitlines()
        for part in (wrap(line, width=90) or [""])
    ]
    if not lines:
        return
    figure = plt.figure(figsize=(10, max(3, min(10, 1.2 + 0.3 * len(lines)))))
    figure.canvas.manager.set_window_title(title)
    figure.text(0.03, 0.97, "\n".join(lines), va="top", family="monospace")
    figure.suptitle(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.show()
    plt.close(figure)


def load_gray(name: str) -> np.ndarray:
    image = cv2.imread(str(BASE_DIR / name), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read {BASE_DIR / name}")
    return image


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (reference.shape[1], reference.shape[0]))


def save_panels(filename: str, panels: list[tuple[str, np.ndarray]]) -> None:
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def main() -> None:
    first = load_gray("cameraman.tif")
    second = resize_like(load_gray("pout.tif"), first)
    result = cv2.subtract(first, second)

    save_panels(
        "solve4_image_subtraction.png",
        [("Image A", first), ("Image B", second), ("A - B (clipped)", result)],
    )
    print(f"Subtraction output range is {result.min()} to {result.max()}.")


if __name__ == "__main__":
    run_with_gui(main, "Problem 4 Results")
