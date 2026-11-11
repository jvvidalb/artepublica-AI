"""Data loading: one subfolder per class under the data directory."""

import random
from collections import defaultdict

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def stratified_split(targets, val_fraction, seed):
    """Split indices so every class keeps roughly val_fraction of its images for validation."""
    by_class = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[target].append(index)

    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for indices in by_class.values():
        rng.shuffle(indices)
        # Keep at least one validation image per class, unless the class has a single image.
        n_val = max(1, round(len(indices) * val_fraction)) if len(indices) > 1 else 0
        val_indices += indices[:n_val]
        train_indices += indices[n_val:]
    return train_indices, val_indices


def make_loaders(data_dir, batch_size=32, val_fraction=0.2, seed=0, workers=2):
    """Return (train_loader, val_loader, class_names)."""
    train_set = datasets.ImageFolder(data_dir, transform=train_transform)
    val_set = datasets.ImageFolder(data_dir, transform=eval_transform)
    train_indices, val_indices = stratified_split(train_set.targets, val_fraction, seed)

    train_loader = DataLoader(Subset(train_set, train_indices), batch_size=batch_size,
                              shuffle=True, num_workers=workers)
    val_loader = DataLoader(Subset(val_set, val_indices), batch_size=batch_size,
                            shuffle=False, num_workers=workers)
    return train_loader, val_loader, train_set.classes
