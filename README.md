# Image Processing Lab Works (Python)

Python implementations of **Professor Md Abdul Masud's** Image Processing lab tasks using **OpenCV**, **NumPy**, and **Matplotlib**.

This repo contains **8 separate scripts** (`lab1.py` … `lab8.py`) and uses standard sample images:
`cameraman.tif`, `pout.tif`, `trees.tif`, `toycars1.png`, `toycars2.png`.

---

## ✅ Lab Tasks Overview
1. **lab1.py** — Convert color image to grayscale and apply thresholding  
2. **lab2.py** — Logarithmic transformation for contrast enhancement  
3. **lab3.py** — Exponential transformation  
4. **lab4.py** — Histogram plot with **two peaks** (background + foreground separation)  
5. **lab5.py** — Remove salt & pepper + Gaussian noise using **mean filter**  
6. **lab6.py** — Remove salt & pepper + Gaussian noise using **median filter**  
7. **lab7.py** — Remove salt & pepper + Gaussian noise using **rank filter**  
8. **lab8.py** — Remove salt & pepper + Gaussian noise using **Gaussian filter**

---

## 📁 Folder Structure
```text
.
├── lab1.py
├── lab2.py
├── lab3.py
├── lab4.py
├── lab5.py
├── lab6.py
├── lab7.py
├── lab8.py
├── cameraman.tif
├── pout.tif
├── trees.tif
├── toycars1.png
└── toycars2.png
```

---

## 🧰 Requirements
- Python 3.8+
- `opencv-python` — Image processing library
- `numpy` — Numerical computing
- `matplotlib` — Visualization and plotting

Install dependencies:
```bash
pip install opencv-python numpy matplotlib
```

---

## ▶ How to Run

Run any lab file individually:
```bash
python lab1.py
python lab2.py
python lab3.py
```

Each script will display output (images/histograms) using Matplotlib windows.

---

## 📝 Notes
- For `lab1.py`, tune the threshold value `T` inside the script for different results.
- For `lab5.py`–`lab8.py`, scripts add noise first, then apply filters to visualize denoising effects.
- All output images display side-by-side for easy comparison.

---

## 👨‍🏫 Credits
**Instructor:** Professor Md Abdul Masud  
**Author:** Azrul Amaline
