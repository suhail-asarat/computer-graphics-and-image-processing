"""Problem 20: Huffman encode, decode, evaluate, and verify an image."""

from contextlib import redirect_stdout
import heapq
from io import StringIO
from dataclasses import dataclass
from itertools import count
from pathlib import Path
import sys
from textwrap import wrap
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


class Tee(StringIO):
    def write(self, text: str) -> int:
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        return super().write(text)


def run_with_gui(main, title: str) -> None:
    capture = Tee()
    with redirect_stdout(capture):
        main()
    text = "\n".join(
        part for line in capture.getvalue().strip().splitlines()
        for part in (wrap(line, width=90) or [""])
    )
    if text:
        figure = plt.figure(figsize=(10, 6))
        figure.canvas.manager.set_window_title(title)
        figure.text(0.03, 0.95, text, va="top", family="monospace")
        figure.suptitle(title, fontweight="bold")
        plt.axis("off")
        plt.show()
        plt.close(figure)


def load_gray(name: str) -> np.ndarray:
    image = cv2.imread(str(BASE_DIR / name), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read {BASE_DIR / name}")
    return image


def entropy(image: np.ndarray) -> float:
    histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / image.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


@dataclass
class Node:
    symbol: Optional[int] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None


def build_tree(frequencies: np.ndarray) -> Node:
    serial = count()
    heap: list[tuple[int, int, Node]] = [
        (int(frequency), next(serial), Node(symbol=symbol))
        for symbol, frequency in enumerate(frequencies)
        if frequency > 0
    ]
    heapq.heapify(heap)
    if len(heap) == 1:
        return heap[0][2]
    while len(heap) > 1:
        frequency_a, _, left = heapq.heappop(heap)
        frequency_b, _, right = heapq.heappop(heap)
        parent = Node(left=left, right=right)
        heapq.heappush(
            heap, (frequency_a + frequency_b, next(serial), parent)
        )
    return heap[0][2]


def make_codes(
    node: Node, prefix: str = "", codes: Optional[dict[int, str]] = None
) -> dict[int, str]:
    if codes is None:
        codes = {}
    if node.symbol is not None:
        codes[node.symbol] = prefix or "0"
        return codes
    make_codes(node.left, prefix + "0", codes)
    make_codes(node.right, prefix + "1", codes)
    return codes


def decode(bit_string: str, tree: Node, count_to_decode: int) -> np.ndarray:
    if tree.symbol is not None:
        return np.full(count_to_decode, tree.symbol, dtype=np.uint8)
    decoded = np.empty(count_to_decode, dtype=np.uint8)
    output_index = 0
    node = tree
    for bit in bit_string:
        node = node.left if bit == "0" else node.right
        if node.symbol is not None:
            decoded[output_index] = node.symbol
            output_index += 1
            if output_index == count_to_decode:
                break
            node = tree
    if output_index != count_to_decode:
        raise ValueError("Encoded data ended before all pixels were decoded.")
    return decoded


def main() -> None:
    image = load_gray("cameraman.tif")
    frequencies = np.bincount(image.ravel(), minlength=256)
    tree = build_tree(frequencies)
    codes = make_codes(tree)

    encoded = "".join(codes[int(pixel)] for pixel in image.ravel())
    bit_array = np.fromiter((bit == "1" for bit in encoded), dtype=np.uint8)
    packed = np.packbits(bit_array)

    decoded = decode(encoded, tree, image.size).reshape(image.shape)
    perfect_reconstruction = np.array_equal(image, decoded)
    assert perfect_reconstruction

    original_bits = image.size * 8
    compressed_bits = len(encoded)
    average_code_length = compressed_bits / image.size
    compression_ratio = original_bits / compressed_bits
    report = (
        f"Unique symbols: {len(codes)}\n"
        f"Entropy: {entropy(image):.4f} bits/pixel\n"
        f"Average Huffman length: {average_code_length:.4f} bits/pixel\n"
        f"Original size: {original_bits} bits\n"
        f"Compressed data size: {compressed_bits} bits\n"
        f"Packed in-memory size: {packed.nbytes} bytes\n"
        f"Compression ratio: {compression_ratio:.4f}:1\n"
        f"Perfect reconstruction: {perfect_reconstruction}"
    )
    print(report)

    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[1].imshow(decoded, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Huffman reconstruction")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    run_with_gui(main, "Problem 20 Results")
