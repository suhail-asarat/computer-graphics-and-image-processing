"""Problem 19: Measure spatial, physical, and informational image properties."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from textwrap import wrap

import cv2
import matplotlib.pyplot as plt
import numpy as np

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
        figure = plt.figure(figsize=(10, 8))
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


def main() -> None:
    image_path = BASE_DIR / "cameraman.tif"
    image = load_gray(image_path.name)
    height, width = image.shape
    channels = 1
    bit_depth = image.dtype.itemsize * 8
    pixel_count = image.size
    uncompressed_bits = pixel_count * channels * bit_depth
    image_entropy = entropy(image)
    estimated_information_bits = image_entropy * pixel_count
    actual_file_bytes = Path(image_path).stat().st_size
    dynamic_range = int(image.max()) - int(image.min())

    report = (
        "SPATIAL PROPERTIES\n"
        f"Width x height: {width} x {height} pixels\n"
        f"Total pixels: {pixel_count}\n"
        f"Channels: {channels}\n\n"
        "PHYSICAL/STORAGE PROPERTIES\n"
        f"Data type: {image.dtype}\n"
        f"Bit depth: {bit_depth} bits/pixel\n"
        f"Theoretical uncompressed size: {uncompressed_bits} bits "
        f"({uncompressed_bits / 8:.0f} bytes)\n"
        f"Actual file size: {actual_file_bytes} bytes\n\n"
        "INFORMATIONAL PROPERTIES\n"
        f"Minimum intensity: {image.min()}\n"
        f"Maximum intensity: {image.max()}\n"
        f"Dynamic range: {dynamic_range}\n"
        f"Mean intensity: {np.mean(image):.4f}\n"
        f"Standard deviation: {np.std(image):.4f}\n"
        f"Entropy: {image_entropy:.4f} bits/pixel\n"
        f"Estimated information: {estimated_information_bits:.2f} bits\n"
    )
    print(report)


if __name__ == "__main__":
    run_with_gui(main, "Problem 19 Results")
