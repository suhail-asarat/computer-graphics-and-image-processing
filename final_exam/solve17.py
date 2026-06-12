"""Problem 17: Apply order-25 (maximum) filtering with a 5x5 window."""

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


def add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    noise = np.random.default_rng(42).normal(0, 20, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray) -> np.ndarray:
    random_values = np.random.default_rng(42).random(image.shape)
    noisy = image.copy()
    noisy[random_values < 0.025] = 0
    noisy[random_values > 0.975] = 255
    return noisy


def save_panels(filename: str, panels: list[tuple[str, np.ndarray]]) -> None:
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(figure)


def maximum_filter_5x5(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), dtype=np.uint8)
    return cv2.dilate(image, kernel)


def main() -> None:
    original = load_gray("cameraman.tif")
    gaussian_noisy = add_gaussian_noise(original)
    salt_pepper_noisy = add_salt_pepper_noise(original)
    gaussian_filtered = maximum_filter_5x5(gaussian_noisy)
    salt_pepper_filtered = maximum_filter_5x5(salt_pepper_noisy)

    save_panels(
        "solve17_order25_max_filter.png",
        [
            ("Original", original),
            ("Gaussian noise", gaussian_noisy),
            ("5x5 order-25 max", gaussian_filtered),
            ("Salt-pepper noise", salt_pepper_noisy),
            ("5x5 order-25 max", salt_pepper_filtered),
        ],
    )
    assert np.all(gaussian_filtered >= gaussian_noisy)
    assert np.all(salt_pepper_filtered >= salt_pepper_noisy)
    print("The 25th ranked value in each 5x5 neighborhood was selected.")


if __name__ == "__main__":
    main()
