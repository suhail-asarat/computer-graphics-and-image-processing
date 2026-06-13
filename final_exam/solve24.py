"""Problem 24: Train and evaluate a CNN on the MNIST digit database."""

import argparse
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import sys
from textwrap import wrap

import matplotlib

import matplotlib.pyplot as plt
import numpy as np

# Codex sets DEBUG=release, while tinygrad expects DEBUG to be an integer.
os.environ["DEBUG"] = "0"

from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist

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
        figure = plt.figure(figsize=(10, 5))
        figure.canvas.manager.set_window_title(title)
        figure.text(0.03, 0.95, text, va="top", family="monospace")
        figure.suptitle(title, fontweight="bold")
        plt.axis("off")
        plt.show()
        plt.close(figure)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=5000,
        help="Use 0 for all 60,000 training images.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=1000,
        help="Use 0 for all 10,000 test images.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def prepare_data(limit: int, images: Tensor, labels: Tensor):
    count = len(images) if limit <= 0 else min(limit, len(images))
    image_array = images[:count].numpy().astype(np.float32) / 255.0
    image_array = (image_array - 0.1307) / 0.3081
    label_array = labels[:count].numpy().astype(np.int32)
    return image_array, label_array


def predict_in_batches(
    model: DigitCNN, images: np.ndarray, batch_size: int
) -> np.ndarray:
    predictions = []
    for start in range(0, len(images), batch_size):
        logits = model(Tensor(images[start : start + batch_size]))
        predictions.append(logits.argmax(axis=1).numpy())
    return np.concatenate(predictions)


def main() -> None:
    args = parse_args()
    np.random.seed(42)
    Tensor.manual_seed(42)

    train_images, train_labels, test_images, test_labels = mnist()
    x_train, y_train = prepare_data(args.train_limit, train_images, train_labels)
    x_test, y_test = prepare_data(args.test_limit, test_images, test_labels)

    model = DigitCNN()
    optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=0.001)

    for epoch in range(args.epochs):
        permutation = np.random.permutation(len(x_train))
        total_loss = 0.0
        with Tensor.train():
            for start in range(0, len(x_train), args.batch_size):
                indices = permutation[start : start + args.batch_size]
                images = Tensor(x_train[indices])
                labels = Tensor(y_train[indices])
                loss = model(images).sparse_categorical_crossentropy(labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(indices)
        print(
            f"Epoch {epoch + 1}/{args.epochs}, "
            f"loss={total_loss / len(x_train):.4f}"
        )

    with Tensor.train(False):
        predictions = predict_in_batches(model, x_test, args.batch_size)
    accuracy = float(np.mean(predictions == y_test) * 100)

    figure, axes = plt.subplots(2, 5, figsize=(10, 4))
    for axis, image, label, prediction in zip(
        axes.ravel(), x_test[:10], y_test[:10], predictions[:10]
    ):
        axis.imshow(image.squeeze(), cmap="gray")
        axis.set_title(f"True {label}, Pred {prediction}")
        axis.axis("off")
    figure.suptitle(f"MNIST CNN predictions - accuracy {accuracy:.2f}%")
    figure.tight_layout()
    plt.show()
    plt.close(figure)

    assert len(predictions) == len(y_test)
    print("Framework: tinygrad")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Test accuracy: {accuracy:.2f}%")
    print("The trained model remains in memory and was not saved to disk.")


if __name__ == "__main__":
    run_with_gui(main, "Problem 24 Results")
