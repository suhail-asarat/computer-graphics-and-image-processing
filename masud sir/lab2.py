import cv2
import numpy as np
import matplotlib.pyplot as plt

I = cv2.imread("cameraman.tif", cv2.IMREAD_GRAYSCALE)
if I is None:
    raise FileNotFoundError("cameraman.tif not found")

r = I.astype(np.float32) / 255.0

# s = c * log(1 + r), choose c so max becomes 1
c = 1.0 / np.log(1.0 + np.max(r))
s = c * np.log(1.0 + r)

J = (np.clip(s, 0, 1) * 255).astype(np.uint8)

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1); plt.imshow(I, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(J, cmap="gray"); plt.title("Log Transform"); plt.axis("off")
plt.tight_layout()
plt.show()