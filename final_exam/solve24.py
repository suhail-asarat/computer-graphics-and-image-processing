"""Problem 24: Train and evaluate a CNN on the MNIST digit database."""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Codex sets DEBUG=release, while tinygrad expects DEBUG to be an integer.
os.environ["DEBUG"] = "0"

from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


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

    model_path = OUTPUT_DIR / "solve24_mnist_cnn.safetensors"
    nn.state.safe_save(nn.state.get_state_dict(model), str(model_path))

    figure, axes = plt.subplots(2, 5, figsize=(10, 4))
    for axis, image, label, prediction in zip(
        axes.ravel(), x_test[:10], y_test[:10], predictions[:10]
    ):
        axis.imshow(image.squeeze(), cmap="gray")
        axis.set_title(f"True {label}, Pred {prediction}")
        axis.axis("off")
    figure.suptitle(f"MNIST CNN predictions - accuracy {accuracy:.2f}%")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "solve24_mnist_predictions.png", dpi=150)
    plt.close(figure)

    assert len(predictions) == len(y_test)
    print("Framework: tinygrad")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
