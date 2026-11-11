"""ResNet-50 with a frozen ImageNet-pretrained backbone and a new classification head."""

from torch import nn
from torchvision import models


def build_model(num_classes, pretrained=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    # The replacement head is the only trainable part.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
