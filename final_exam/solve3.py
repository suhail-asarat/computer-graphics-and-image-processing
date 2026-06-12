"""Problem 3: Add a constant value to every image pixel."""

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
    axes = [axes] if len(panels) == 1 else axes
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(figure)


def main() -> None:
    image = load_gray("cameraman.tif")
    constant = 50
    result = np.clip(image.astype(np.int16) + constant, 0, 255).astype(np.uint8)

    save_panels(
        "solve3_add_constant.png",
        [("Original", image), (f"Original + {constant}", result)],
    )
    assert result.dtype == np.uint8
    print(f"Added {constant}; output range is {result.min()} to {result.max()}.")


if __name__ == "__main__":
    main()
