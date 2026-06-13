"""Problem 10: Apply exponential intensity transformation."""

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
    text = "\n".join(
        part
        for line in capture.getvalue().strip().splitlines()
        for part in (wrap(line, width=90) or [""])
    )
    if text:
        figure = plt.figure(figsize=(10, 5))
        figure.canvas.manager.set_window_title(title)
        figure.text(0.03, 0.95, text, va="top", family="monospace")
        figure.suptitle(title, fontweight="bold")
        plt.axis("off")
        plt.show()
        plt.close(figure)


def load_gray(name: str) -> np.ndarray:
    image = cv2.imread(str(BASE_DIR / name), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read {BASE_DIR / name}")
    return image


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
    image = load_gray("cameraman.tif")
    normalized = image.astype(np.float32) / 255.0
    base = 1.6
    transformed = ((np.power(base, normalized) - 1.0) / (base - 1.0) * 255)
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)

    save_panels(
        "solve10_exponential_transform.png",
        [("Original", image), (f"Exponential, base={base}", transformed)],
    )
    print(f"Exponential transformation completed with base={base}.")


if __name__ == "__main__":
    run_with_gui(main, "Problem 10 Results")
