
import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_pepper(img_u8, amount=0.02):
    noisy = img_u8.copy()
    h, w = noisy.shape
    n = int(amount * h * w)

    # salt
    ys = np.random.randint(0, h, n // 2)
    xs = np.random.randint(0, w, n // 2)
    noisy[ys, xs] = 255

    # pepper
    ys = np.random.randint(0, h, n // 2)
    xs = np.random.randint(0, w, n // 2)
    noisy[ys, xs] = 0

    return noisy

def add_gaussian_noise(img_u8, sigma=20):
    noise = np.random.normal(0, sigma, img_u8.shape).astype(np.float32)
    out = img_u8.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

I = cv2.imread("cameraman.tif", 0)
if I is None:
    raise FileNotFoundError("cameraman.tif not found")

noisy = add_salt_pepper(I, amount=0.03)
noisy = add_gaussian_noise(noisy, sigma=20)

# Mean filter (average blur)
den = cv2.blur(noisy, (5, 5))

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1); plt.imshow(I, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(noisy, cmap="gray"); plt.title("Noisy (S&P + Gaussian)"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(den, cmap="gray"); plt.title("Mean Filter (5x5)"); plt.axis("off")
plt.tight_layout()
plt.show()