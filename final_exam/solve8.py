"""Problem 8: Perform image thresholding."""

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
    image = load_gray("cameraman.tif")
    manual_threshold = 128
    _, manual = cv2.threshold(image, manual_threshold, 255, cv2.THRESH_BINARY)
    otsu_threshold, otsu = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    save_panels(
        "solve8_thresholding.png",
        [
            ("Original", image),
            (f"Manual T={manual_threshold}", manual),
            (f"Otsu T={otsu_threshold:.0f}", otsu),
        ],
    )
    print(f"Manual threshold={manual_threshold}; Otsu threshold={otsu_threshold:.0f}.")


if __name__ == "__main__":
    main()
