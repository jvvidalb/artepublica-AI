"""Train a new classification head on top of a pretrained ResNet-50.

Usage: python src/train.py --data data/raw --epochs 10
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from dataset import make_loaders
from model import build_model


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            correct += (predictions == labels).sum().item()
            total += len(labels)
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw", help="folder with one subfolder per class")
    parser.add_argument("--out", default="models/classifier.pt", help="checkpoint path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, classes = make_loaders(
        args.data, args.batch_size, args.val_fraction, args.seed)
    print(f"{len(classes)} classes, {len(train_loader.dataset)} training images, "
          f"{len(val_loader.dataset)} validation images, device={device}")

    model = build_model(len(classes)).to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        # The frozen backbone stays in eval mode so its batch-norm statistics are not disturbed.
        model.eval()
        model.fc.train()
        running_loss = seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            seen += len(labels)

        accuracy = evaluate(model, val_loader, device)
        print(f"epoch {epoch}/{args.epochs}  loss {running_loss / seen:.4f}  val accuracy {accuracy:.3f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({"state_dict": model.state_dict(), "classes": classes}, out_path)

    print(f"best validation accuracy {best_accuracy:.3f}, checkpoint saved to {out_path}")


if __name__ == "__main__":
    main()
