import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader

from layers import Conv2d, ReLU


def Classifier() -> nn.Module:
    def block(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            Conv2d(in_channels, out_channels, 3, 2, 1),
            ReLU(),
        )

    return nn.Sequential(
        block(3, 32),
        block(32, 64),
        block(64, 128),
        block(128, 256),
        Conv2d(256, 10, kernel_size=2, stride=1, padding=0),
        Rearrange("b c h w -> b (c h w)"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_name", type=str)
    args = parser.parse_args()

    num_epochs = 100
    batch_size = 128
    learning_rate = 3e-4
    device = torch.device("cuda")
    log_dir = Path("runs") / args.training_name
    log_dir.mkdir(parents=True, exist_ok=False)

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_transform = transforms.Compose(
        [
            test_transform,
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.094, 0.094)),
        ]
    )

    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
    )

    model = Classifier().to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)

    with open(log_dir / "log.csv", "w") as log_file:
        fieldnames = [
            "epoch",
            "time",
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
        ]
        writer = csv.DictWriter(log_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()

        start = time.perf_counter()
        for epoch_idx in range(num_epochs):
            loss_sum = 0.0
            num_samples = 0
            num_correct = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = F.cross_entropy(input=logits, target=labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                num_samples += logits.shape[0]
                num_correct += (logits.argmax(dim=1) == labels).sum().item()

            train_loss = loss_sum / len(train_loader)
            train_acc = num_correct / num_samples

            with torch.no_grad():
                loss_sum = 0.0
                num_samples = 0
                num_correct = 0

                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model(images)
                    loss = F.cross_entropy(input=logits, target=labels)

                    loss_sum += loss.item()
                    num_samples += logits.shape[0]
                    num_correct += (logits.argmax(dim=1) == labels).sum().item()

                test_loss = loss_sum / len(test_loader)
                test_acc = num_correct / num_samples

            writer.writerow(
                {
                    "epoch": epoch_idx,
                    "time": time.perf_counter() - start,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )
            log_file.flush()


if __name__ == "__main__":
    main()
