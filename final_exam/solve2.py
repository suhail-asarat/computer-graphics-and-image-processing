"""Problem 2: Compute a 32-bin histogram for an 8-bit image."""

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
    bin_width = 256 // 32
    bin_indices = image.ravel() // bin_width
    histogram = np.bincount(bin_indices, minlength=32)
    starts = np.arange(32) * bin_width

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(starts, histogram, width=bin_width, align="edge", edgecolor="black")
    ax.set(
        title="32-bin histogram (8 intensity levels per bin)",
        xlabel="Starting intensity",
        ylabel="Pixel count",
        xlim=(0, 256),
    )
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    assert histogram.size == 32 and histogram.sum() == image.size
    print("32-bin histogram:", histogram.tolist())


if __name__ == "__main__":
    run_with_gui(main, "Problem 2 Results")
