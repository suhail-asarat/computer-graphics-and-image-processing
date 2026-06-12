"""Problem 18: Compare Gaussian filters with sigma=1 and sigma=3."""

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


def sharpness(image: np.ndarray) -> float:
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def main() -> None:
    original = load_gray("cameraman.tif")
    gaussian_noisy = add_gaussian_noise(original)
    salt_pepper_noisy = add_salt_pepper_noise(original)

    g_sigma1 = cv2.GaussianBlur(gaussian_noisy, (0, 0), sigmaX=1)
    g_sigma3 = cv2.GaussianBlur(gaussian_noisy, (0, 0), sigmaX=3)
    sp_sigma1 = cv2.GaussianBlur(salt_pepper_noisy, (0, 0), sigmaX=1)
    sp_sigma3 = cv2.GaussianBlur(salt_pepper_noisy, (0, 0), sigmaX=3)

    save_panels(
        "solve18_gaussian_sigma_comparison.png",
        [
            ("Gaussian noise", gaussian_noisy),
            ("Gaussian, sigma=1", g_sigma1),
            ("Gaussian, sigma=3", g_sigma3),
            ("Salt-pepper noise", salt_pepper_noisy),
            ("Gaussian, sigma=1", sp_sigma1),
            ("Gaussian, sigma=3", sp_sigma3),
        ],
    )
    gaussian_scores = [sharpness(x) for x in (gaussian_noisy, g_sigma1, g_sigma3)]
    salt_pepper_scores = [
        sharpness(x) for x in (salt_pepper_noisy, sp_sigma1, sp_sigma3)
    ]
    assert gaussian_scores[2] < gaussian_scores[1]
    print("Laplacian-variance sharpness, Gaussian noise:", gaussian_scores)
    print("Laplacian-variance sharpness, salt-pepper noise:", salt_pepper_scores)
    print("Higher sigma removes more noise but also reduces image sharpness.")


if __name__ == "__main__":
    main()
