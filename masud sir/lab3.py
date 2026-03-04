import cv2
import numpy as np
import matplotlib.pyplot as plt

I = cv2.imread("cameraman.tif", cv2.IMREAD_GRAYSCALE)
if I is None:
    raise FileNotFoundError("cameraman.tif not found")

r = I.astype(np.float32) / 255.0

# s = ( (1+a)^r - 1 ), then normalize
a = 0.4
s = ((1.0 + a) ** r) - 1.0
s = s / (s.max() + 1e-8)

J = (np.clip(s, 0, 1) * 255).astype(np.uint8)

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1); plt.imshow(I, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(J, cmap="gray"); plt.title("Exponential Transform"); plt.axis("off")
plt.tight_layout()
plt.show()