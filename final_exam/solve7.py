"""Problem 7: Convert two images to binary and perform XOR."""

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


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
    figure.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(figure)


def main() -> None:
    first = load_gray("cameraman.tif")
    second = resize_like(load_gray("pout.tif"), first)
    threshold_a, binary_a = cv2.threshold(
        first, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold_b, binary_b = cv2.threshold(
        second, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    result = cv2.bitwise_xor(binary_a, binary_b)

    save_panels(
        "solve7_binary_xor.png",
        [("Binary A", binary_a), ("Binary B", binary_b), ("A XOR B", result)],
    )
    assert set(result.ravel()).issubset({0, 255})
    print(f"Otsu thresholds: A={threshold_a:.0f}, B={threshold_b:.0f}.")


if __name__ == "__main__":
    main()
