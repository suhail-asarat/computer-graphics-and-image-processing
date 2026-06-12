"""Problem 5: Perform image multiplication and division."""

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
    first = load_gray("cameraman.tif").astype(np.float32)
    second = resize_like(load_gray("pout.tif"), first).astype(np.float32)

    multiplied = np.clip(first * second / 255.0, 0, 255).astype(np.uint8)
    divided = np.clip((first / (second + 1.0)) * 128.0, 0, 255).astype(np.uint8)

    save_panels(
        "solve5_multiplication_division.png",
        [
            ("Image A", first),
            ("Image B", second),
            ("A x B / 255", multiplied),
            ("128 x A / (B + 1)", divided),
        ],
    )
    print("Image multiplication and protected division completed.")


if __name__ == "__main__":
    main()
