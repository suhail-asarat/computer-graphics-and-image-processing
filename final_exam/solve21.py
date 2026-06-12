"""Problem 21: Compare first- and second-order derivative edge detectors."""

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


def main() -> None:
    image = load_gray("cameraman.tif")
    float_image = image.astype(np.float32)

    sobel_x = cv2.Sobel(float_image, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(float_image, cv2.CV_32F, 0, 1, ksize=3)
    first_order = np.hypot(sobel_x, sobel_y)

    laplacian = cv2.Laplacian(float_image, cv2.CV_32F, ksize=3)
    second_order = np.abs(laplacian)

    first_edges = (first_order > np.percentile(first_order, 85)).astype(
        np.uint8
    ) * 255
    second_edges = (second_order > np.percentile(second_order, 85)).astype(
        np.uint8
    ) * 255

    save_panels(
        "solve21_derivative_edge_comparison.png",
        [
            ("Original", image),
            ("1st order: Sobel magnitude", normalize_u8(first_order)),
            ("1st order edges", first_edges),
            ("2nd order: |Laplacian|", normalize_u8(second_order)),
            ("2nd order edges", second_edges),
        ],
    )
    print(
        "Detected edge pixels - first order:",
        int(np.count_nonzero(first_edges)),
        "second order:",
        int(np.count_nonzero(second_edges)),
    )
    print("Sobel gives gradient strength; Laplacian reacts to rapid changes in all directions.")


if __name__ == "__main__":
    main()
