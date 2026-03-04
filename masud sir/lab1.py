import cv2
import numpy as np
import matplotlib.pyplot as plt

# Use a color image (toycars are color)
img_bgr = cv2.imread("toycars2.png", cv2.IMREAD_COLOR)
if img_bgr is None:
    raise FileNotFoundError("toycars2.png not found")

# Convert to grayscale
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Thresholding (manual threshold; change T if needed)
T = 120
binary = np.where(gray >= T, 255, 0).astype(np.uint8)

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); plt.title("Color"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(gray, cmap="gray"); plt.title("Grayscale"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(binary, cmap="gray"); plt.title(f"Threshold (T={T})"); plt.axis("off")
plt.tight_layout()
plt.show()