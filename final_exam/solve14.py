"""Problem 14: Match the histogram of one image to a reference image."""

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
        part for line in capture.getvalue().strip().splitlines()
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


def histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source_hist = np.bincount(source.ravel(), minlength=256).astype(np.float64)
    reference_hist = np.bincount(
        reference.ravel(), minlength=256
    ).astype(np.float64)
    source_cdf = np.cumsum(source_hist) / source.size
    reference_cdf = np.cumsum(reference_hist) / reference.size

    lookup = np.empty(256, dtype=np.uint8)
    for intensity in range(256):
        lookup[intensity] = np.argmin(
            np.abs(reference_cdf - source_cdf[intensity])
        )
    return lookup[source]


def main() -> None:
    source = load_gray("pout.tif")
    reference = resize_like(load_gray("cameraman.tif"), source)
    matched = histogram_match(source, reference)
    save_panels(
        "solve14_histogram_matching.png",
        [
            ("Source", source),
            ("Reference", reference),
            ("Histogram matched", matched),
        ],
    )
    assert matched.shape == source.shape
    print("Source histogram matched to the cameraman reference.")


if __name__ == "__main__":
    run_with_gui(main, "Problem 14 Results")
