"""ImageNet-pretrained networks with a frozen backbone and a new classification head."""

from torch import nn
from torchvision import models

ARCHITECTURES = ("resnet50", "mobilenet_v2")


def build_model(arch, num_classes, pretrained=True):
    """Return (model, head): the whole network and its trainable final layer."""
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    elif arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    else:
        raise ValueError(f"unknown architecture {arch!r}, expected one of {ARCHITECTURES}")

    for param in model.parameters():
        param.requires_grad = False

    # The replacement head is the only trainable part.
    if arch == "resnet50":
        head = nn.Linear(model.fc.in_features, num_classes)
        model.fc = head
    else:
        head = nn.Linear(model.classifier[1].in_features, num_classes)
        model.classifier[1] = head
    return model, head
