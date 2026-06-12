"""Problem 2: Compute a 32-bin histogram for an 8-bit image."""

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
    fig.savefig(OUTPUT_DIR / "solve2_histogram_32_bins.png", dpi=150)
    plt.close(fig)

    assert histogram.size == 32 and histogram.sum() == image.size
    print("32-bin histogram:", histogram.tolist())


if __name__ == "__main__":
    main()
