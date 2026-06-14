# Image Processing Lab Solutions

This repository contains solutions for the Image Processing Lab problems based on the final exam list.

## Section 02: Image Transformation

### Problem 1: Histogram of an 8-bit Grayscale Image
**Explanation:** The problem requires calculating and displaying the frequency of each intensity value (0-255) in an 8-bit grayscale image.
**Theory & Logic:** A histogram represents the distribution of pixel intensities in an image. By counting the occurrences of each intensity level, we can visualize the image's contrast and brightness. We use `numpy.bincount` to quickly count the occurrences of each pixel intensity up to 256.
**Code Snippet:**
```python
image = load_gray("cameraman.tif")
histogram = np.bincount(image.ravel(), minlength=256)
```

### Problem 2: 32-Bin Histogram
**Explanation:** Compute a histogram with 32 bins for an image with 256 intensity levels.
**Theory & Logic:** Instead of counting each of the 256 levels individually, we group them into 32 bins. Each bin covers 256 / 32 = 8 intensity levels. We achieve this by applying integer division to the pixel values by 8 before counting their occurrences.
**Code Snippet:**
```python
bin_width = 256 // 32
bin_indices = image.ravel() // bin_width
histogram = np.bincount(bin_indices, minlength=32)
```

### Problem 3: Add Constant Value
**Explanation:** Add a constant value to every pixel in an image to increase its overall brightness.
**Theory & Logic:** Adding a constant directly impacts the intensity of the image. To avoid overflow (where values exceed 255 and wrap around to 0), we must temporarily cast the image to a wider integer type, perform the addition, clip the results to the [0, 255] range, and then convert back to 8-bit unsigned integers.
**Code Snippet:**
```python
constant = 50
result = np.clip(image.astype(np.int16) + constant, 0, 255).astype(np.uint8)
```

### Problem 4: Image Subtraction
**Explanation:** Subtract one image from another.
**Theory & Logic:** Subtraction highlights the differences between two images and is often used for detecting changes or background subtraction. OpenCV's `cv2.subtract` seamlessly handles the subtraction while automatically clipping negative values to zero to remain within a valid 8-bit range.
**Code Snippet:**
```python
first = load_gray("cameraman.tif")
second = resize_like(load_gray("pout.tif"), first)
result = cv2.subtract(first, second)
```

### Problem 5: Image Multiplication and Division
**Explanation:** Perform pixel-wise multiplication and division between two images.
**Theory & Logic:** Multiplication can act as an image mask, while division can be used to normalize illumination. Both operations require floating-point arithmetic to maintain numeric precision. We normalize the multiplier by 255.0 to keep the result to scale, and for division, we add a small constant (1.0) to the denominator to prevent division-by-zero errors.
**Code Snippet:**
```python
multiplied = np.clip(first * second / 255.0, 0, 255).astype(np.uint8)
divided = np.clip((first / (second + 1.0)) * 128.0, 0, 255).astype(np.uint8)
```

### Problem 6: Image Inversion
**Explanation:** Create a negative version of an image.
**Theory & Logic:** Inverting an image swaps darks and lights, similar to a photographic negative film. For an 8-bit image, the pixel's inverse is cleanly obtained by subtracting its original intensity from the maximum possible value (255).
**Code Snippet:**
```python
inverted = 255 - image
```

### Problem 7: Binary Image XOR
**Explanation:** Convert two images to binary and perform a bitwise XOR operation.
**Theory & Logic:** We first convert the grayscale images to binary (0 and 255) using Otsu's thresholding, which automatically discovers an optimal threshold value. Then, we apply the bitwise XOR operation, which sets a pixel to 255 if the corresponding pixels in the input images differ, and 0 if they match.
**Code Snippet:**
```python
_, binary_a = cv2.threshold(first, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, binary_b = cv2.threshold(second, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
result = cv2.bitwise_xor(binary_a, binary_b)
```

### Problem 8: Image Thresholding
**Explanation:** Segment an image into foreground and background using a threshold.
**Theory & Logic:** Thresholding binarizes a grayscale image. Pixels exceeding the threshold are set to the maximum value (255); others are set to 0. A manual threshold sets a hard limit, while Otsu's method dynamically determines a threshold based on the image's overall bimodal histogram.
**Code Snippet:**
```python
manual_threshold = 128
_, manual = cv2.threshold(image, manual_threshold, 255, cv2.THRESH_BINARY)
otsu_threshold, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### Problem 9: Logarithmic Transformation
**Explanation:** Apply a logarithmic transformation to expose details in dark regions.
**Theory & Logic:** The log transformation formula expands the discrete values of dark pixels while simultaneously compressing the higher-level values. It effectively enhances details in darker regions of an image. We multiply by a scaling constant to ensure the maximum transformed value maps to 255.
**Code Snippet:**
```python
constant = 255.0 / np.log1p(float(image.max()))
transformed = (constant * np.log1p(image.astype(np.float32))).astype(np.uint8)
```

### Problem 10: Exponential Transformation
**Explanation:** Apply an exponential transformation to enhance bright details.
**Theory & Logic:** Exponential transformation expands bright pixel values and compresses dark ones. We normalize the image values to the [0, 1] range before applying the exponent operation, then rescale back to the 0-255 limit.
**Code Snippet:**
```python
normalized = image.astype(np.float32) / 255.0
base = 1.6
transformed = ((np.power(base, normalized) - 1.0) / (base - 1.0) * 255)
transformed = np.clip(transformed, 0, 255).astype(np.uint8)
```

### Problem 11: Histogram-Based Image Quality
**Explanation:** Differentiate a poor-quality image from a good one using its histogram.
**Theory & Logic:** A low-contrast image features a narrow, clustered histogram, whereas a high-contrast (good) image possesses a spread-out histogram traversing the complete dynamic range. We extract the 5th and 95th percentiles to determine the effective histogram spread objectively.
**Code Snippet:**
```python
p05, p95 = np.percentile(image, [5, 95])
histogram_spread = float(p95 - p05)
```

### Problem 12: Contrast Stretching
**Explanation:** Enhance contrast by stretching intensities between the 5th and 95th percentiles.
**Theory & Logic:** Contrast stretching linearly remaps the target intensity range so that the 5th percentile anchors to 0, and the 95th percentile caps at 255. This successfully disregards anomalous outliers that might otherwise restrict the usable dynamic range.
**Code Snippet:**
```python
histogram = np.bincount(image.ravel(), minlength=256)
cumulative = np.cumsum(histogram) / image.size
c = int(np.searchsorted(cumulative, 0.05))
d = int(np.searchsorted(cumulative, 0.95))
stretched = np.clip((image.astype(np.float32) - c) * 255.0 / max(d - c, 1), 0, 255).astype(np.uint8)
```

### Problem 13: Histogram Equalization
**Explanation:** Spread pixel intensities uniformly using histogram equalization.
**Theory & Logic:** Histogram equalization treats the cumulative distribution function (CDF) of the image's histogram as a mapping curve. Passing the image through this curve enforces a flat, uniform histogram, vastly improving global contrast.
**Code Snippet:**
```python
histogram = np.bincount(image.ravel(), minlength=256)
cumulative = histogram.cumsum()
nonzero = cumulative[cumulative > 0]
cdf_min = nonzero[0]
lookup = np.round((cumulative - cdf_min) * 255.0 / max(image.size - cdf_min, 1))
lookup = np.clip(lookup, 0, 255).astype(np.uint8)
result = lookup[image]
```

### Problem 14: Histogram Matching
**Explanation:** Adjust an image so its histogram matches a provided reference image.
**Theory & Logic:** We compile the cumulative distribution functions (CDFs) for both the source and reference images. A lookup table is built that maps each source intensity to the reference intensity sharing the closest CDF percentage.
**Code Snippet:**
```python
source_cdf = np.cumsum(source_hist) / source.size
reference_cdf = np.cumsum(reference_hist) / reference.size
lookup = np.empty(256, dtype=np.uint8)
for intensity in range(256):
    lookup[intensity] = np.argmin(np.abs(reference_cdf - source_cdf[intensity]))
matched = lookup[source]
```

## Section 03: Image Enhancement

### Problem 1: 3x3 Mean Filter
**Explanation:** Apply a 3x3 mean filter to denoise an image.
**Theory & Logic:** A mean filter sets each pixel to the average value of its local 3x3 neighborhood. It performs as a low-pass filter, blurring the image to soften Gaussian noise at the cost of diminished edge sharpness.
**Code Snippet:**
```python
gaussian_filtered = cv2.blur(gaussian_noisy, (3, 3))
salt_pepper_filtered = cv2.blur(salt_pepper_noisy, (3, 3))
```

### Problem 2: 3x3 Median Filter
**Explanation:** Use a 3x3 median filter to eliminate noise from an image.
**Theory & Logic:** The median filter assigns each pixel the median statistical value from its 3x3 neighborhood. It perfectly neutralizes salt-and-pepper (impulse) noise while preserving sharp boundaries significantly better than standard mean filters.
**Code Snippet:**
```python
gaussian_filtered = cv2.medianBlur(gaussian_noisy, 3)
salt_pepper_filtered = cv2.medianBlur(salt_pepper_noisy, 3)
```

### Problem 3: Order Filtering (Max Filter)
**Explanation:** Process an image with a 5x5 order-25 (maximum) filter.
**Theory & Logic:** An order-statistic filter chooses an intensity based on its sorted rank within a window. In a 5x5 grid, the 25th element represents the local maximum. This operates identically to grayscale morphological dilation, expanding bright regions and masking dark noise.
**Code Snippet:**
```python
def maximum_filter_5x5(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), dtype=np.uint8)
    return cv2.dilate(image, kernel)
```

### Problem 4: Gaussian Filter Standard Deviation
**Explanation:** Observe how varying the standard deviation (sigma) impacts Gaussian blurring.
**Theory & Logic:** A Gaussian filter uses a normal distribution curve to assign spatial blur weights. Scaling up the standard deviation mathematically widens the bell curve, catching a broader region which removes more noise but causes a heavy loss in overall image sharpness.
**Code Snippet:**
```python
g_sigma1 = cv2.GaussianBlur(gaussian_noisy, (0, 0), sigmaX=1)
g_sigma3 = cv2.GaussianBlur(gaussian_noisy, (0, 0), sigmaX=3)
```

## Section 04: Image Compression

### Problem 1: Image Properties
**Explanation:** Evaluate and calculate the spatial, physical, and informational traits of an image.
**Theory & Logic:** Extracting image traits requires pulling the dimension axes for spatial data, mapping the structural bit depth for theoretical storage bounds, and applying Shannon's entropy formula to measure informational density per pixel.
**Code Snippet:**
```python
def entropy(image: np.ndarray) -> float:
    histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / image.size
    return float(-np.sum(probabilities * np.log2(probabilities)))
```

### Problem 2: Huffman Coding Compression
**Explanation:** Utilize Huffman coding to losslessly compress and then reconstruct an image.
**Theory & Logic:** Huffman coding analyzes symbol frequencies to generate an optimal prefix tree, allocating shorter binary codes to common intensities and longer codes to rare ones. The pixel array is collapsed into this custom binary string for space efficiency and later accurately reconstructed traversing the tree.
**Code Snippet:**
```python
tree = build_tree(frequencies)
codes = make_codes(tree)
encoded = "".join(codes[int(pixel)] for pixel in image.ravel())
```

## Section 05: Image Segmentation

### Problem 1: Derivative Operators for Edge Detection
**Explanation:** Gauge and contrast first-order and second-order derivatives for capturing edges.
**Theory & Logic:** First-order (Sobel) derivatives measure the raw gradient magnitude, often producing thicker edges. Second-order (Laplacian) derivatives compute the rate of gradient change to locate zero-crossings, exposing much finer edges but suffering from heightened noise vulnerability.
**Code Snippet:**
```python
sobel_x = cv2.Sobel(float_image, cv2.CV_32F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(float_image, cv2.CV_32F, 0, 1, ksize=3)
first_order = np.hypot(sobel_x, sobel_y)
laplacian = cv2.Laplacian(float_image, cv2.CV_32F, ksize=3)
second_order = np.abs(laplacian)
```

### Problem 2: Isolated Point Detection
**Explanation:** Target and detect isolated pixel points via a specific Laplacian mask.
**Theory & Logic:** A structured 3x3 Laplacian mask featuring an intense central peak (+8) bordered by deep negatives (-1) responds aggressively to standalone points (drastic omnidirectional variance) while collapsing to zero in flat textures. We apply a threshold over the filter's output to strictly segment the points.
**Code Snippet:**
```python
LAPLACIAN_POINT_MASK = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
response = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, LAPLACIAN_POINT_MASK)
detections = (np.abs(response) >= threshold).astype(np.uint8) * 255
```

### Problem 3: Roberts, Prewitt, and Sobel Operators
**Explanation:** Implement the Roberts, Prewitt, and Sobel kernels for detecting gradient edge magnitudes.
**Theory & Logic:** Roberts detects diagonal gradients using small 2x2 blocks. Prewitt employs 3x3 separated uniform kernels handling broad horizontal/vertical edges. Sobel mirrors Prewitt but injects a central weight bias to lightly smooth noise prior to gradient estimation, producing a stronger structural edge map.
**Code Snippet:**
```python
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = sobel_x.T
sobel_magnitude, sobel_direction = gradient(image, sobel_x, sobel_y)
```

## Section 06: Image Pattern Classification

### Problem 1: CNN for MNIST Digit Classification
**Explanation:** Assemble and train a Convolutional Neural Network (CNN) to correctly categorize MNIST handwritten digits.
**Theory & Logic:** The architecture sequentially stacks convolutional layers for robust spatial feature extraction, relies on max pooling for resolution downsampling, and terminates in fully connected linear layers to compress the features into final digit predictions. The Adam optimizer is utilized to actively lower the categorical cross-entropy error throughout training.
**Code Snippet:**
```python
class DigitCNN:
    def __init__(self) -> None:
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def __call__(self, images: Tensor) -> Tensor:
        features = self.conv1(images).relu().max_pool2d()
        features = self.conv2(features).relu().max_pool2d()
        features = features.reshape(features.shape[0], -1)
        features = self.fc1(features).relu().dropout(0.25)
        return self.fc2(features)
```
