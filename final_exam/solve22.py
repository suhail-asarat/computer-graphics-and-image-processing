"""Problem 22: Detect isolated points using a Laplacian point mask."""

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


LAPLACIAN_POINT_MASK = np.array(
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32
)


def detect_points(
    image: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    response = cv2.filter2D(
        image.astype(np.float32), cv2.CV_32F, LAPLACIAN_POINT_MASK
    )
    detections = (np.abs(response) >= threshold).astype(np.uint8) * 255
    return response, detections


def main() -> None:
    synthetic = np.full((128, 128), 100, dtype=np.uint8)
    known_point = (64, 64)
    synthetic[known_point] = 255
    response, detected = detect_points(synthetic, threshold=500)
    assert detected[known_point] == 255

    image = load_gray("cameraman.tif")
    real_response, real_detected = detect_points(image, threshold=500)
    save_panels(
        "solve22_isolated_point_detection.png",
        [
            ("Known isolated point", synthetic),
            ("Laplacian response", normalize_u8(np.abs(response))),
            ("Verified detection", detected),
            ("Test image", image),
            ("Image point candidates", real_detected),
        ],
    )
    print(
        f"Known point at {known_point} detected successfully; "
        f"{np.count_nonzero(real_detected)} candidates found in the test image."
    )
    print(f"Maximum test-image response: {np.max(np.abs(real_response)):.2f}.")


if __name__ == "__main__":
    main()
