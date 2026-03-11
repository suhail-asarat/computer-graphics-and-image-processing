import cv2
import numpy as np 
import matplotlib.pyplot as plt

I= cv2.imread("cameraman.tif" , cv2.IMREAD_GRAYSCALE)

if I is None:
    raise FileNotFoundError("Cameraman.tif is not found")

constant = 50  
result = np.clip(I.astype(np.int16) + constant, 0, 255).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(I, cmap='gray'); plt.title('Original Image'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(result, cmap='gray'); plt.title(f'Image after adding {constant}'); plt.axis('off')
plt.tight_layout()
plt.show()