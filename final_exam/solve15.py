"""Problem 15: Apply a 3x3 mean filter to Gaussian and salt-pepper noise."""

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


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    mse = np.mean((reference.astype(float) - test.astype(float)) ** 2)
    return float("inf") if mse == 0 else float(20 * np.log10(255 / np.sqrt(mse)))


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
    original = load_gray("cameraman.tif")
    gaussian_noisy = add_gaussian_noise(original)
    salt_pepper_noisy = add_salt_pepper_noise(original)
    gaussian_filtered = cv2.blur(gaussian_noisy, (3, 3))
    salt_pepper_filtered = cv2.blur(salt_pepper_noisy, (3, 3))

    save_panels(
        "solve15_mean_filter.png",
        [
            ("Original", original),
            ("Gaussian noise", gaussian_noisy),
            ("3x3 mean (Gaussian)", gaussian_filtered),
            ("Salt-pepper noise", salt_pepper_noisy),
            ("3x3 mean (S&P)", salt_pepper_filtered),
        ],
    )
    print(
        "Gaussian PSNR:",
        f"{psnr(original, gaussian_noisy):.2f} -> "
        f"{psnr(original, gaussian_filtered):.2f} dB",
    )
    print(
        "Salt-pepper PSNR:",
        f"{psnr(original, salt_pepper_noisy):.2f} -> "
        f"{psnr(original, salt_pepper_filtered):.2f} dB",
    )


if __name__ == "__main__":
    main()
