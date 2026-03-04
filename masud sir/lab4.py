import cv2
import numpy as np
import matplotlib.pyplot as plt

I = cv2.imread("trees.tif", cv2.IMREAD_GRAYSCALE)
if I is None:
    raise FileNotFoundError("trees.tif not found")

hist = np.bincount(I.ravel(), minlength=256).astype(np.float64)

# Smooth histogram a bit to detect peaks
kernel = np.ones(9) / 9.0
h_smooth = np.convolve(hist, kernel, mode="same")

# Find local maxima
peaks = []
for i in range(1, 255):
    if h_smooth[i] > h_smooth[i - 1] and h_smooth[i] > h_smooth[i + 1]:
        peaks.append(i)

# Take top 2 peaks by height
peaks = sorted(peaks, key=lambda x: h_smooth[x], reverse=True)[:2]
peaks = sorted(peaks)  # for nicer display

plt.figure(figsize=(10, 4))
plt.plot(hist, label="Histogram")
plt.plot(h_smooth, label="Smoothed", linewidth=2)
for p in peaks:
    plt.axvline(p, linestyle="--")
    plt.text(p + 2, h_smooth[p], f"Peak @ {p}", fontsize=10)
plt.title("Bimodal Histogram (Two Peaks)")
plt.xlim([0, 255])
plt.grid(True)
plt.legend()
plt.show()