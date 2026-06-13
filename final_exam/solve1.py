"""Problem 1: Display the histogram of an 8-bit grayscale image."""

from contextlib import redirect_stdout
from io import StringIO
import sys
from textwrap import wrap

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

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


def main() -> None:
    image = load_gray("cameraman.tif")
    histogram = np.bincount(image.ravel(), minlength=256)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("8-bit grayscale image")
    axes[0].axis("off")
    axes[1].plot(np.arange(256), histogram, color="black")
    axes[1].set(title="Histogram", xlabel="Intensity", ylabel="Pixel count")
    axes[1].set_xlim(0, 255)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    assert histogram.sum() == image.size
    print(f"Histogram contains {histogram.sum()} pixels across 256 levels.")


if __name__ == "__main__":
    run_with_gui(main, "Problem 1 Results")
