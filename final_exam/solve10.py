"""Problem 10: Apply exponential intensity transformation."""

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
    normalized = image.astype(np.float32) / 255.0
    base = 1.6
    transformed = ((np.power(base, normalized) - 1.0) / (base - 1.0) * 255)
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)

    save_panels(
        "solve10_exponential_transform.png",
        [("Original", image), (f"Exponential, base={base}", transformed)],
    )
    print(f"Exponential transformation completed with base={base}.")


if __name__ == "__main__":
    main()
