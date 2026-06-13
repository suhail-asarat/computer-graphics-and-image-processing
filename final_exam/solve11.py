"""Problem 11: Differentiate low-contrast and good images by histogram spread."""

from contextlib import redirect_stdout
from io import StringIO
import sys
from textwrap import wrap

import cv2
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
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


def entropy(image: np.ndarray) -> float:
    histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / image.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def quality_metrics(image: np.ndarray) -> dict[str, float]:
    p05, p95 = np.percentile(image, [5, 95])
    return {
        "standard deviation": float(image.std()),
        "90% histogram spread": float(p95 - p05),
        "entropy": entropy(image),
    }


def main() -> None:
    poor = load_gray("pout.tif")
    good = cv2.equalizeHist(poor)
    poor_metrics = quality_metrics(poor)
    good_metrics = quality_metrics(good)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].imshow(poor, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Low-contrast image")
    axes[0, 1].imshow(good, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Contrast-enhanced image")
    for axis in axes[0]:
        axis.axis("off")
    axes[1, 0].hist(poor.ravel(), bins=256, range=(0, 256), color="black")
    axes[1, 0].set_title("Narrow histogram = poor contrast")
    axes[1, 1].hist(good.ravel(), bins=256, range=(0, 256), color="black")
    axes[1, 1].set_title("Wide histogram = good contrast")
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    assert (
        good_metrics["90% histogram spread"]
        > poor_metrics["90% histogram spread"]
    )
    print("Low-contrast metrics:", poor_metrics)
    print("Good-image metrics:", good_metrics)


if __name__ == "__main__":
    run_with_gui(main, "Problem 11 Results")
