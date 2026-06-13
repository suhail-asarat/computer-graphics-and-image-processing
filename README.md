# Computer Graphics and Image Processing Lab

Python implementations of image-processing laboratory problems using OpenCV,
NumPy, Matplotlib, and tinygrad.

The repository includes the original laboratory exercises and 24 standalone
final-exam solutions based on:

`final_exam/Lab Problems in Image Processing.docx`

## Repository Structure

```text
.
|-- .venv/
|-- exam/
|-- final_exam/
|   |-- solve1.py ... solve24.py
|   |-- requirements.txt
|   |-- cameraman.tif
|   |-- pout.tif
|   |-- trees.tif
|   |-- toycars1.png
|   |-- toycars2.png
|   `-- outputs/
|-- masud sir/
|   |-- lab1.py ... lab8.py
|   `-- sample images
`-- README.md
```

Every `solveN.py` file is independent. There is no shared `common.py` module,
so an individual solution can be studied, submitted, or executed by itself.

## Final Exam Problems

### Section 02: Image Transformation

| File | Problem |
|---|---|
| `solve1.py` | Histogram of an 8-bit grayscale image |
| `solve2.py` | Histogram with 32 bins and 256 intensity levels |
| `solve3.py` | Add a constant value to every pixel |
| `solve4.py` | Subtract one image from another |
| `solve5.py` | Image multiplication and division |
| `solve6.py` | Image inversion or negative transformation |
| `solve7.py` | Binary conversion and XOR operation |
| `solve8.py` | Manual and Otsu thresholding |
| `solve9.py` | Logarithmic transformation |
| `solve10.py` | Exponential transformation |
| `solve11.py` | Compare poor and good images using histograms |
| `solve12.py` | Contrast stretching using 5th and 95th percentiles |
| `solve13.py` | Histogram equalization |
| `solve14.py` | Histogram matching |

### Section 03: Image Enhancement

| File | Problem |
|---|---|
| `solve15.py` | 3x3 mean filtering for Gaussian and salt-pepper noise |
| `solve16.py` | 3x3 median filtering for Gaussian and salt-pepper noise |
| `solve17.py` | Order-25 maximum filtering with a 5x5 window |
| `solve18.py` | Gaussian filtering with sigma values 1 and 3 |

### Section 04: Image Compression

| File | Problem |
|---|---|
| `solve19.py` | Spatial, physical, and informational image properties |
| `solve20.py` | Huffman encoding, decoding, compression, and reconstruction |

### Section 05: Image Segmentation

| File | Problem |
|---|---|
| `solve21.py` | First-order and second-order derivative edge detection |
| `solve22.py` | Isolated-point detection using a Laplacian mask |
| `solve23.py` | Roberts, Prewitt, and Sobel edge detection |

### Section 06: Image Pattern Classification

| File | Problem |
|---|---|
| `solve24.py` | CNN classification of MNIST handwritten digits |

## Requirements

- Python 3.14 or a compatible recent Python version
- OpenCV
- NumPy
- Matplotlib
- tinygrad, used by the MNIST CNN

Install the final-exam dependencies into the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\final_exam\requirements.txt
```

## Running Solutions

Run any solution from the repository root:

```powershell
.\.venv\Scripts\python.exe .\final_exam\solve1.py
.\.venv\Scripts\python.exe .\final_exam\solve15.py
.\.venv\Scripts\python.exe .\final_exam\solve23.py
```

Each script finds its sample images relative to its own location. It does not
depend on the terminal's current directory.

To run solutions 1 through 23 in PowerShell:

```powershell
1..23 | ForEach-Object {
    .\.venv\Scripts\python.exe ".\final_exam\solve$_.py"
}
```

## Running the MNIST CNN

The default command trains for one epoch using 5,000 training images and
evaluates 1,000 test images:

```powershell
.\.venv\Scripts\python.exe .\final_exam\solve24.py
```

The training size, test size, batch size, and epoch count can be changed:

```powershell
.\.venv\Scripts\python.exe .\final_exam\solve24.py `
    --epochs 2 `
    --train-limit 10000 `
    --test-limit 2000 `
    --batch-size 64
```

Use `--train-limit 0 --test-limit 0` to use the complete MNIST dataset.
The dataset is downloaded automatically the first time the script runs.

## Generated Results

All generated figures, reports, compressed data, reconstructed images, and the
trained CNN model are saved in:

```text
final_exam/outputs/
```

Examples include:

- Histogram and transformation comparison figures
- Noise-filtering comparisons and PSNR measurements
- Image-property and Huffman compression reports
- Huffman codebook, compressed data, and reconstructed image
- Edge-detection comparison figures
- MNIST prediction figure and trained CNN model

The scripts use Matplotlib's non-interactive backend, so they save results
without requiring GUI windows.

## Verification

All 24 standalone final-exam scripts were executed successfully using the
project virtual environment. Huffman decoding reproduced the original image
exactly. The default-sized MNIST CNN test reached approximately 86% accuracy
after one epoch; results can vary slightly between runs and configurations.

## Credits

**Instructor:** Professor Md Abdul Masud  
**Author:** Azrul Amaline
