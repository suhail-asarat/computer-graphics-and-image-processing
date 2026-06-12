"""Problem 23: Apply Roberts, Prewitt, and Sobel edge detectors."""

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


def normalize_u8(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def save_panels(filename: str, panels: list[tuple[str, np.ndarray]]) -> None:
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(figure)


def gradient(
    image: np.ndarray, kernel_x: np.ndarray, kernel_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gx = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel_y)
    magnitude = np.hypot(gx, gy)
    direction = np.degrees(np.arctan2(gy, gx))
    return magnitude, direction


def main() -> None:
    image = load_gray("cameraman.tif")
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    prewitt_x = np.array([[-1, 0, 1]] * 3, dtype=np.float32)
    prewitt_y = prewitt_x.T
    sobel_x = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32
    )
    sobel_y = sobel_x.T

    roberts_magnitude, _ = gradient(image, roberts_x, roberts_y)
    prewitt_magnitude, _ = gradient(image, prewitt_x, prewitt_y)
    sobel_magnitude, sobel_direction = gradient(image, sobel_x, sobel_y)

    direction_for_display = np.zeros_like(image)
    strong_edges = sobel_magnitude > np.percentile(sobel_magnitude, 70)
    direction_for_display[strong_edges] = (
        (sobel_direction[strong_edges] + 180) / 360 * 255
    ).astype(np.uint8)
    save_panels(
        "solve23_roberts_prewitt_sobel.png",
        [
            ("Original", image),
            ("Roberts magnitude", normalize_u8(roberts_magnitude)),
            ("Prewitt magnitude", normalize_u8(prewitt_magnitude)),
            ("Sobel magnitude", normalize_u8(sobel_magnitude)),
            ("Sobel direction", direction_for_display),
        ],
    )
    print(f"Maximum Roberts gradient: {roberts_magnitude.max():.2f}")
    print(f"Maximum Prewitt gradient: {prewitt_magnitude.max():.2f}")
    print(f"Maximum Sobel gradient: {sobel_magnitude.max():.2f}")
    print(
        "Sobel direction range:",
        f"{sobel_direction.min():.2f} to {sobel_direction.max():.2f} degrees.",
    )


if __name__ == "__main__":
    main()
