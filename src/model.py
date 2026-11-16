"""ImageNet-pretrained networks with a frozen backbone and a new classification head."""

from torch import nn
from torchvision import models

ARCHITECTURES = ("resnet50", "mobilenet_v2", "efficientnet_b0", "mobilenetv4")

# MobileNetV4 is not in torchvision; this is the medium convolutional variant from timm.
MOBILENETV4_TIMM_NAME = "mobilenetv4_conv_medium.e500_r256_in1k"


def build_model(arch, num_classes, pretrained=True):
    """Return (model, head): the whole network and its trainable final layer."""
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    elif arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    elif arch == "mobilenetv4":
        import timm  # imported here so the torchvision networks work without timm installed
        model = timm.create_model(MOBILENETV4_TIMM_NAME, pretrained=pretrained)
    else:
        raise ValueError(f"unknown architecture {arch!r}, expected one of {ARCHITECTURES}")

    for param in model.parameters():
        param.requires_grad = False

    # The replacement head is the only trainable part.
    if arch == "resnet50":
        head = nn.Linear(model.fc.in_features, num_classes)
        model.fc = head
    elif arch == "mobilenetv4":
        model.reset_classifier(num_classes)
        head = model.get_classifier()
    else:
        head = nn.Linear(model.classifier[1].in_features, num_classes)
        model.classifier[1] = head
    return model, head
