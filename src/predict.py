"""Classify new images with a trained checkpoint.

Usage: python src/predict.py --checkpoint models/resnet50.pt path/to/image.jpg [more images...]
"""

import argparse

import torch

from dataset import eval_transform, load_image
from model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="image files to classify")
    parser.add_argument("--checkpoint", default="models/resnet50.pt")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint["classes"]
    model, _ = build_model(checkpoint["arch"], len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    top_k = min(args.top_k, len(classes))
    for path in args.images:
        image = eval_transform(load_image(path)).unsqueeze(0)
        with torch.no_grad():
            probabilities = model(image).softmax(dim=1)[0]
        scores, indices = probabilities.topk(top_k)
        print(path)
        for score, index in zip(scores, indices):
            print(f"  {classes[index]:<30} {score.item():.1%}")


if __name__ == "__main__":
    main()
