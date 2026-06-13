"""Problem 12: Contrast stretch using 5th and 95th histogram percentiles."""

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
    image = load_gray("pout.tif")
    histogram = np.bincount(image.ravel(), minlength=256)
    cumulative = np.cumsum(histogram) / image.size
    c = int(np.searchsorted(cumulative, 0.05))
    d = int(np.searchsorted(cumulative, 0.95))

    stretched = (image.astype(np.float32) - c) * 255.0 / max(d - c, 1)
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    save_panels(
        "solve12_contrast_stretch.png",
        [("Original", image), (f"Stretched (c={c}, d={d})", stretched)],
    )
    print(f"5th percentile c={c}; 95th percentile d={d}.")


if __name__ == "__main__":
    run_with_gui(main, "Problem 12 Results")
