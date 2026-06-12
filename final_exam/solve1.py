"""Problem 1: Display the histogram of an 8-bit grayscale image."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


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
    fig.savefig(OUTPUT_DIR / "solve1_histogram.png", dpi=150)
    plt.close(fig)

    assert histogram.sum() == image.size
    print(f"Histogram contains {histogram.sum()} pixels across 256 levels.")


if __name__ == "__main__":
    main()
