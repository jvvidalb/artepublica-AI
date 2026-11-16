"""Generate notebooks/artepublica_colab.ipynb.

Usage: python scripts/build_notebook.py

The notebook is self-contained so it can be uploaded to Google Colab on its own.
Edit the cells below and re-run this script instead of editing the .ipynb by hand.
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

CELLS = []


def md(text):
    CELLS.append(new_markdown_cell(text.strip("\n")))


def code(text):
    CELLS.append(new_code_cell(text.strip("\n")))


# --------------------------------------------------------------------------- overview
md(r"""
# Arte Pública — recognising public sculptures with pretrained networks

This notebook trains and compares four ImageNet-pretrained networks on photos of public
sculptures in Rio de Janeiro. For each network the pretrained backbone is **frozen** and only
a new final layer (one output per sculpture) is trained. This is transfer learning by feature
extraction, which suits a small dataset.

| Network | Source | Why it is here |
| --- | --- | --- |
| ResNet-50 | torchvision | standard, strong baseline |
| MobileNetV2 | torchvision | small, phone-friendly (2018) |
| EfficientNet-B0 | torchvision | small, more accurate than MobileNetV2 on ImageNet (2019) |
| MobileNetV4 (conv-medium) | timm | current generation of mobile networks (2024) |

## How to use it

1. **Turn on the GPU:** menu *Runtime → Change runtime type → T4 GPU → Save*.
2. **Give it the images** (details in section 3). The easiest way: on GitHub open the repository,
   pick the branch that holds `data/raw`, choose *Code → Download ZIP*, then drag that zip into
   Colab's *Files* panel (folder icon on the left) and wait for the upload to finish.
3. **Run everything:** menu *Runtime → Run all*.
4. **Send back the results:** the last cell downloads `artepublica_results.zip`
   (a `results.json` plus the figures). That file is what the report is written from.

Every section below starts with a short note on what the cell does and what to look for.
""")

# --------------------------------------------------------------------------- 1 setup
md(r"""
## 1. Setup

Installs `timm` (needed only for MobileNetV4), imports the libraries and reports which device
will be used.

**Look for:** `device: cuda` and a GPU name. If it says `cpu`, the notebook still works but is
much slower; enable the GPU as described above and run again.
""")

code(r"""
%pip install -q timm

import json
import platform
import random
import shutil
import subprocess
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torchvision
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else platform.processor() or "cpu"

print(f"device: {DEVICE} ({DEVICE_NAME})")
print(f"torch {torch.__version__}, torchvision {torchvision.__version__}, timm {timm.__version__}")
""")

# --------------------------------------------------------------------------- 2 config
md(r"""
## 2. Configuration

All the settings of the experiment are in this one cell.

- `ARCHS` — the networks to compare. Remove a name to skip that network.
- `EPOCHS` — passes over the training images per run.
- `SEEDS` — each seed is a separate run with a **different random train/validation split**.
  With about 34 validation images per split, one run is noisy; several seeds give an average
  and a spread.
- `VAL_FRACTION` — share of each class held out for validation.

The defaults mean 4 networks × 3 seeds = 12 short training runs.
""")

code(r"""
ARCHS = ["resnet50", "mobilenet_v2", "efficientnet_b0", "mobilenetv4"]
EPOCHS = 20
SEEDS = [0, 1, 2]
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.2

# Where the images come from if they are not already in the Colab working folder.
REPO_URL = "https://github.com/jvvidalb/artepublica-AI.git"
REPO_BRANCH = "claude/pensive-wright-7427we"

# Everything the notebook produces goes here.
OUTPUT_DIR = Path("artepublica_results")
CHECKPOINT_DIR = Path("artepublica_checkpoints")
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
""")

# --------------------------------------------------------------------------- 3 data
md(r"""
## 3. Get the images

The images must end up in a folder laid out as `data/raw/<sculpture name>/<photos>`: the folder
name is the label. This cell looks for them in this order and stops at the first that works:

1. **Already here** — a `data/raw` folder in the working directory (for example from an
   earlier run of this cell).
2. **A GitHub zip in the working folder** — the *Code → Download ZIP* file from GitHub,
   dragged into Colab's *Files* panel. It is about 240 MB, so let the upload finish first
   (the progress ring at the bottom of the Files panel disappears).
3. **`git clone`** — works without any upload, but only if the repository is public.
4. **Upload dialog** — as a last resort the cell asks you to pick the GitHub zip. This is the
   slowest route.

**Look for:** the table of classes at the end. Each sculpture should have a plausible number
of photos, and there should be no class you do not recognise.
""")

code(r"""
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_data_dir():
    # Return the first data/raw folder that contains class subfolders, or None.
    for root in (Path("."), Path("..")):
        direct = root / "data" / "raw"
        candidates = [direct] if direct.is_dir() else []
        if root == Path("."):
            candidates += sorted(root.glob("*/data/raw")) + sorted(root.glob("*/*/data/raw"))
        for candidate in candidates:
            if candidate.is_dir() and any(child.is_dir() for child in candidate.iterdir()):
                return candidate
    return None


def extract_github_zip():
    # Unpack the first zip in the working folder that contains a data/raw folder.
    for zip_path in sorted(Path(".").glob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                if any("/data/raw/" in name for name in archive.namelist()):
                    print(f"extracting {zip_path.name} ...")
                    archive.extractall("github_zip")
                    return True
        except zipfile.BadZipFile:
            print(f"{zip_path.name} is not a complete zip (upload still running?)")
    return False


DATA_DIR = find_data_dir()

if DATA_DIR is None and extract_github_zip():
    DATA_DIR = find_data_dir()

if DATA_DIR is None:
    print("no images found locally, trying git clone ...")
    clone = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, "artepublica-AI"],
        capture_output=True, text=True)
    if clone.returncode == 0:
        DATA_DIR = find_data_dir()
    else:
        print("git clone failed (the repository is probably private).")

if DATA_DIR is None and IN_COLAB:
    print("Choose the zip downloaded from GitHub (Code -> Download ZIP).")
    colab_files.upload()
    if extract_github_zip():
        DATA_DIR = find_data_dir()

if DATA_DIR is None:
    raise SystemExit("No data/raw folder found. Put the GitHub zip in the working folder and run this cell again.")

CLASS_NAMES = sorted(child.name for child in DATA_DIR.iterdir() if child.is_dir())
SAMPLES = []  # (path, class index)
for class_index, class_name in enumerate(CLASS_NAMES):
    for path in sorted((DATA_DIR / class_name).iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            SAMPLES.append((path, class_index))

print(f"\nimages folder: {DATA_DIR}")
print(f"{'class':<34}{'images':>7}")
for class_index, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name:<34}{sum(1 for _, label in SAMPLES if label == class_index):>7}")
print(f"{'total':<34}{len(SAMPLES):>7}")
""")

# --------------------------------------------------------------------------- 4 load + look
md(r"""
## 4. Load the images and look at them

The photos are large phone pictures, and decoding them is the slowest part of training. This
cell reads each one **once**, applies the rotation recorded by the phone (EXIF), shrinks it so
the long side is at most 512 px (the networks only see 224 px crops) and keeps it in memory.

It then shows a few photos per class.

**Look for:** photos the right way up, and every photo in a row showing the sculpture named on
the left. A photo in the wrong row is a labelling mistake to fix in `data/raw`.
""")

code(r"""
CACHE_SIZE = 512


def load_image(path):
    # Open as RGB, apply the EXIF rotation, and shrink to at most CACHE_SIZE on the long side.
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((CACHE_SIZE, CACHE_SIZE))
        return image


started = time.time()
IMAGES = []
for number, (path, _) in enumerate(SAMPLES, start=1):
    IMAGES.append(load_image(path))
    if number % 25 == 0 or number == len(SAMPLES):
        print(f"loaded {number}/{len(SAMPLES)} images", end="\r")
LABELS = [label for _, label in SAMPLES]
print(f"\nloaded {len(IMAGES)} images in {time.time() - started:.0f} s")

PER_ROW = 6
figure, axes = plt.subplots(len(CLASS_NAMES), PER_ROW, figsize=(2.2 * PER_ROW, 2.2 * len(CLASS_NAMES)))
picker = random.Random(0)
for row, class_name in enumerate(CLASS_NAMES):
    indices = [index for index, label in enumerate(LABELS) if label == row]
    chosen = picker.sample(indices, min(PER_ROW, len(indices)))
    for column in range(PER_ROW):
        axis = axes[row][column]
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        if column < len(chosen):
            axis.imshow(IMAGES[chosen[column]])
        if column == 0:
            axis.set_ylabel(class_name.replace("_", "\n"), rotation=0, ha="right", va="center", fontsize=9)
figure.suptitle("Random sample of photos per class", y=1.0)
figure.tight_layout()
figure.savefig(OUTPUT_DIR / "sample_images.png", dpi=110, bbox_inches="tight")
plt.show()
""")

# --------------------------------------------------------------------------- 5 pipeline
md(r"""
## 5. Data pipeline

Defines how photos are turned into network inputs and how they are split.

- **Training photos** get a random crop and a random horizontal flip each time they are used
  (data augmentation), so the network never sees exactly the same picture twice.
- **Validation photos** always get the same centre crop, so scores are repeatable.
- **The split is stratified:** every sculpture keeps the same share of its photos for
  validation. The split depends only on the seed, so all four networks are scored on exactly
  the same validation photos for a given seed.

All networks use the same 224 px input so the comparison is like for like. (MobileNetV4 was
pretrained at 256 px; it is run at 224 px here like the others.)

Nothing is printed here; the functions are used in section 7.
""")

code(r"""
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


class SculptureDataset(Dataset):
    # A subset of the in-memory images, with one transform.
    def __init__(self, indices, transform):
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        return self.transform(IMAGES[index]), LABELS[index], index


def stratified_split(seed):
    # Return (train indices, validation indices), keeping VAL_FRACTION of every class for validation.
    by_class = defaultdict(list)
    for index, label in enumerate(LABELS):
        by_class[label].append(index)
    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for label in sorted(by_class):
        indices = by_class[label]
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * VAL_FRACTION)) if len(indices) > 1 else 0
        val_indices += indices[:n_val]
        train_indices += indices[n_val:]
    return train_indices, val_indices


def make_loaders(seed):
    train_indices, val_indices = stratified_split(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(SculptureDataset(train_indices, train_transform), batch_size=BATCH_SIZE,
                              shuffle=True, generator=shuffle_generator, num_workers=2)
    val_loader = DataLoader(SculptureDataset(val_indices, eval_transform), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)
    return train_loader, val_loader
""")

# --------------------------------------------------------------------------- 6 models
md(r"""
## 6. The four pretrained networks

`build_model` downloads a network with its ImageNet weights, freezes every weight, and swaps
the final ImageNet layer (1000 classes) for a new one with one output per sculpture. Only that
new layer is trained.

The cell then builds each network once to report its size.

**Look for:** the table. *Total parameters* is the size of the whole network (what you would
ship in an app); *trained parameters* is the tiny part learned from your photos.
""")

code(r"""
MOBILENETV4_TIMM_NAME = "mobilenetv4_conv_medium.e500_r256_in1k"


def build_model(arch, num_classes):
    # Return (model, head): the whole network and its trainable final layer.
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif arch == "mobilenetv4":
        model = timm.create_model(MOBILENETV4_TIMM_NAME, pretrained=True)
    else:
        raise ValueError(f"unknown architecture {arch!r}")

    for param in model.parameters():
        param.requires_grad = False

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


MODEL_INFO = {}
print(f"{'network':<18}{'total parameters':>18}{'trained parameters':>20}{'feature size':>14}")
for arch in ARCHS:
    model, head = build_model(arch, len(CLASS_NAMES))
    MODEL_INFO[arch] = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trained_parameters": sum(p.numel() for p in head.parameters()),
        "feature_size": head.in_features,
    }
    info = MODEL_INFO[arch]
    print(f"{arch:<18}{info['total_parameters']:>18,}{info['trained_parameters']:>20,}{info['feature_size']:>14}")
    del model, head
""")

# --------------------------------------------------------------------------- 7 training loop
md(r"""
## 7. Training and evaluation functions

- `evaluate` runs the validation photos through a network and returns the loss, the accuracy
  and the prediction for every photo.
- `measure_inference_ms` times how long the network takes to classify one image on this device.
- `run_experiment` does one complete run for one network and one seed: build, train for
  `EPOCHS`, evaluate after every epoch, and return everything worth keeping (per-epoch history,
  final predictions, timings).

Scores are reported for the **final epoch**. The best epoch is recorded too, but picking the
best epoch using the validation photos makes that number slightly optimistic.

Nothing is printed here; the runs happen in the next section.
""")

code(r"""
def evaluate(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, predictions = 0.0, []
    with torch.no_grad():
        for images, labels, indices in loader:
            logits = model(images.to(DEVICE)).float().cpu()
            total_loss += loss_fn(logits, labels).item()
            probabilities = logits.softmax(dim=1)
            confidence, predicted = probabilities.max(dim=1)
            for index, label, guess, sure in zip(indices.tolist(), labels.tolist(), predicted.tolist(), confidence.tolist()):
                predictions.append({"file": f"{CLASS_NAMES[label]}/{SAMPLES[index][0].name}", "index": index,
                                    "true": label, "predicted": guess, "confidence": round(sure, 4)})
    count = len(predictions)
    accuracy = sum(p["true"] == p["predicted"] for p in predictions) / count
    return total_loss / count, accuracy, predictions


def measure_inference_ms(model, repeats=30):
    # Average time to classify one 224 px image on DEVICE, in milliseconds.
    model.eval()
    image = torch.randn(1, 3, 224, 224, device=DEVICE)
    with torch.no_grad():
        for _ in range(5):  # warm-up
            model(image)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            model(image)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats * 1000


def run_experiment(arch, seed):
    torch.manual_seed(seed)
    train_loader, val_loader = make_loaders(seed)
    model, head = build_model(arch, len(CLASS_NAMES))
    model.to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        # The frozen backbone stays in eval mode, so batch-norm statistics and dropout are fixed.
        model.eval()
        running_loss = seen = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            seen += len(labels)

        val_loss, val_accuracy, predictions = evaluate(model, val_loader)
        history.append({"epoch": epoch, "train_loss": round(running_loss / seen, 4),
                        "val_loss": round(val_loss, 4), "val_accuracy": round(val_accuracy, 4),
                        "seconds": round(time.time() - started, 1)})
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  epoch {epoch:>2}/{EPOCHS}  train loss {running_loss / seen:.3f}  "
                  f"val loss {val_loss:.3f}  val accuracy {val_accuracy:.3f}")

    best = max(history, key=lambda row: row["val_accuracy"])
    result = {
        "arch": arch, "seed": seed,
        "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset),
        "final_val_accuracy": history[-1]["val_accuracy"],
        "best_val_accuracy": best["val_accuracy"], "best_epoch": best["epoch"],
        "train_seconds": history[-1]["seconds"],
        "inference_ms_per_image": round(measure_inference_ms(model), 2),
        "history": history,
        "predictions": predictions,  # validation predictions after the final epoch
    }
    return result, model
""")

# --------------------------------------------------------------------------- 8 run
md(r"""
## 8. Run the comparison

Trains every network in `ARCHS` once per seed in `SEEDS`. This is the long cell; on a Colab
T4 GPU expect a few minutes in total.

After each run the results so far are written to `artepublica_results/results.json`, so a
disconnect does not lose finished runs. For each network, the trained weights of the first
seed are saved in `artepublica_checkpoints/` (see section 10 for downloading them).

**Look for:** train loss falling steadily and validation accuracy rising and then levelling
off. If validation accuracy is still climbing at the last epoch, raise `EPOCHS` in section 2.
""")

code(r"""
def save_results():
    payload = {
        "config": {"archs": ARCHS, "epochs": EPOCHS, "seeds": SEEDS, "batch_size": BATCH_SIZE,
                   "learning_rate": LEARNING_RATE, "val_fraction": VAL_FRACTION,
                   "input_size": 224, "frozen_backbone": True,
                   "mobilenetv4_variant": MOBILENETV4_TIMM_NAME},
        "environment": {"device": DEVICE, "device_name": DEVICE_NAME, "in_colab": IN_COLAB,
                        "python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "timm": timm.__version__},
        "dataset": {"classes": CLASS_NAMES,
                    "images_per_class": {name: LABELS.count(i) for i, name in enumerate(CLASS_NAMES)},
                    "total_images": len(SAMPLES)},
        "models": MODEL_INFO,
        "runs": RUNS,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(payload, indent=1))


RUNS = []
experiment_started = time.time()
for arch in ARCHS:
    for seed in SEEDS:
        print(f"{arch}, seed {seed}")
        result, model = run_experiment(arch, seed)
        RUNS.append(result)
        if seed == SEEDS[0]:
            checkpoint_path = CHECKPOINT_DIR / f"{arch}.pt"
            torch.save({"arch": arch, "state_dict": model.state_dict(), "classes": CLASS_NAMES}, checkpoint_path)
            MODEL_INFO[arch]["checkpoint_mb"] = round(checkpoint_path.stat().st_size / 1e6, 1)
        save_results()
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

print(f"\nall {len(RUNS)} runs finished in {(time.time() - experiment_started) / 60:.1f} minutes")
""")

# --------------------------------------------------------------------------- 9 results
md(r"""
## 9. Results

### 9.1 Summary table

One row per network, averaged over the seeds.

- **accuracy (final)** — share of validation photos classified correctly after the last
  epoch: mean, with the lowest and highest seed in brackets. This is the headline number.
- **best epoch** — mean of each run's best validation accuracy (slightly optimistic).
- **params** — total network size in millions of parameters; **file** — checkpoint size.
- **train** — mean time for one full training run; **infer** — time to classify one image on
  this device.

**Look for:** whether the accuracy ranges of two networks overlap. If they do, the data does
not separate them, and size or speed should decide.
""")

code(r"""
def runs_for(arch):
    return [run for run in RUNS if run["arch"] == arch]


SUMMARY = []
for arch in ARCHS:
    runs = runs_for(arch)
    final = [run["final_val_accuracy"] for run in runs]
    SUMMARY.append({
        "arch": arch,
        "final_accuracy_mean": round(float(np.mean(final)), 4),
        "final_accuracy_min": min(final), "final_accuracy_max": max(final),
        "best_epoch_accuracy_mean": round(float(np.mean([run["best_val_accuracy"] for run in runs])), 4),
        "total_parameters": MODEL_INFO[arch]["total_parameters"],
        "checkpoint_mb": MODEL_INFO[arch].get("checkpoint_mb"),
        "train_seconds_mean": round(float(np.mean([run["train_seconds"] for run in runs])), 1),
        "inference_ms_per_image": round(float(np.mean([run["inference_ms_per_image"] for run in runs])), 2),
    })

print(f"{'network':<17}{'accuracy (final)':<24}{'best epoch':>11}{'params':>9}{'file':>9}{'train':>8}{'infer':>10}")
for row in SUMMARY:
    accuracy = f"{row['final_accuracy_mean']:.1%} ({row['final_accuracy_min']:.1%}-{row['final_accuracy_max']:.1%})"
    print(f"{row['arch']:<17}{accuracy:<24}{row['best_epoch_accuracy_mean']:>10.1%} "
          f"{row['total_parameters'] / 1e6:>7.1f}M{row['checkpoint_mb']:>7.1f}MB"
          f"{row['train_seconds_mean']:>7.0f}s{row['inference_ms_per_image']:>8.1f}ms")

(OUTPUT_DIR / "summary.json").write_text(json.dumps(SUMMARY, indent=1))
""")

md(r"""
### 9.2 Learning curves

Left: validation accuracy per epoch. Right: validation loss per epoch. Each line is the mean
over the seeds and the shaded band runs from the lowest to the highest seed.

**Look for:** curves that have flattened by the last epoch (training was long enough), and how
wide the bands are (how much the result depends on the split).
""")

code(r"""
COLORS = {"resnet50": "#0072B2", "mobilenet_v2": "#E69F00", "efficientnet_b0": "#009E73", "mobilenetv4": "#CC79A7"}
epochs = np.arange(1, EPOCHS + 1)

figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for axis, key, title in ((axes[0], "val_accuracy", "Validation accuracy"), (axes[1], "val_loss", "Validation loss")):
    for arch in ARCHS:
        curves = np.array([[row[key] for row in run["history"]] for run in runs_for(arch)])
        color = COLORS.get(arch)
        axis.plot(epochs, curves.mean(axis=0), label=arch, color=color, linewidth=2)
        axis.fill_between(epochs, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.15, linewidth=0)
    axis.set_title(f"{title} (mean and range over {len(SEEDS)} seeds)", fontsize=11)
    axis.set_xlabel("epoch")
    axis.xaxis.get_major_locator().set_params(integer=True)
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("accuracy")
axes[0].set_ylim(top=1.01)
axes[1].set_ylabel("cross-entropy loss")
axes[0].legend(frameon=False, loc="lower right")
figure.tight_layout()
figure.savefig(OUTPUT_DIR / "learning_curves.png", dpi=130, bbox_inches="tight")
plt.show()
""")

md(r"""
### 9.3 Confusion matrices

One grid per network, adding up the validation predictions of all seeds. Rows are the true
sculpture, columns are what the network answered; the diagonal is correct answers.

**Look for:** numbers off the diagonal. They show which sculptures get mistaken for which.
""")

code(r"""
short_names = [name.replace("_", "\n") for name in CLASS_NAMES]
n_classes = len(CLASS_NAMES)
CONFUSION = {}

figure, axes = plt.subplots(1, len(ARCHS), figsize=(4.3 * len(ARCHS), 4.6), squeeze=False)
for axis, arch in zip(axes[0], ARCHS):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for run in runs_for(arch):
        for prediction in run["predictions"]:
            matrix[prediction["true"], prediction["predicted"]] += 1
    CONFUSION[arch] = matrix.tolist()
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.sum(axis=1).max())
    for row in range(n_classes):
        for column in range(n_classes):
            value = matrix[row, column]
            if value:
                dark = value > matrix.sum(axis=1).max() / 2
                axis.text(column, row, value, ha="center", va="center", color="white" if dark else "black")
    axis.set_title(f"{arch}\n{np.trace(matrix)}/{matrix.sum()} correct", fontsize=11)
    axis.set_xticks(range(n_classes), short_names, fontsize=7)
    axis.set_yticks(range(n_classes), short_names if axis is axes[0][0] else [], fontsize=7)
    axis.set_xlabel("predicted")
axes[0][0].set_ylabel("true")
figure.tight_layout()
figure.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=130, bbox_inches="tight")
plt.show()

(OUTPUT_DIR / "confusion_matrices.json").write_text(json.dumps({"classes": CLASS_NAMES, "matrices": CONFUSION}, indent=1))
""")

md(r"""
### 9.4 Accuracy per sculpture

The share of each sculpture's validation photos that each network got right (all seeds added up).

**Look for:** a sculpture that is clearly harder than the others for every network. That
usually points at the photos (too few, too similar to another class) and not at the network.
""")

code(r"""
figure, axis = plt.subplots(figsize=(11, 4))
width = 0.8 / len(ARCHS)
for position, arch in enumerate(ARCHS):
    matrix = np.array(CONFUSION[arch])
    per_class = matrix.diagonal() / matrix.sum(axis=1)
    x = np.arange(n_classes) + (position - (len(ARCHS) - 1) / 2) * width
    bars = axis.bar(x, per_class, width * 0.92, label=arch, color=COLORS.get(arch))
    axis.bar_label(bars, labels=[f"{value:.0%}" for value in per_class], fontsize=7, padding=2)
axis.set_xticks(np.arange(n_classes), [name.replace("_", " ") for name in CLASS_NAMES], fontsize=9)
axis.set_ylabel("validation accuracy")
axis.set_ylim(0, 1.12)
axis.set_title("Accuracy per sculpture", fontsize=11)
axis.spines[["top", "right"]].set_visible(False)
axis.legend(frameon=False, ncol=len(ARCHS), loc="lower center", bbox_to_anchor=(0.5, -0.3))
figure.savefig(OUTPUT_DIR / "per_class_accuracy.png", dpi=130, bbox_inches="tight")
plt.show()
""")

md(r"""
### 9.5 Misclassified photos

Every validation photo that at least one run got wrong, with the true label and which networks
answered what. Photos that several networks miss appear first.

**Look for:** what the misses have in common: sculpture very small in the frame, unusual
angle, something blocking it, or a photo that arguably belongs to another class.
""")

code(r"""
mistakes = defaultdict(list)  # image index -> [(arch, seed, predicted), ...]
for run in RUNS:
    for prediction in run["predictions"]:
        if prediction["true"] != prediction["predicted"]:
            mistakes[prediction["index"]].append((run["arch"], run["seed"], prediction["predicted"]))

print(f"{len(mistakes)} distinct photos were misclassified at least once "
      f"({sum(len(v) for v in mistakes.values())} wrong answers in {len(RUNS)} runs)")

ordered = sorted(mistakes, key=lambda index: -len(mistakes[index]))[:12]
if ordered:
    columns = 4
    rows = (len(ordered) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4.2 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, ordered):
        axis.imshow(IMAGES[index])
        answers = defaultdict(set)
        for arch, _, predicted in mistakes[index]:
            answers[CLASS_NAMES[predicted]].add(arch)
        lines = [f"true: {CLASS_NAMES[LABELS[index]]}"]
        lines += [f"-> {wrong}: {', '.join(sorted(archs))}" for wrong, archs in answers.items()]
        axis.set_title("\n".join(lines), fontsize=8, loc="left")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "misclassified.png", dpi=110, bbox_inches="tight")
    plt.show()
""")

# --------------------------------------------------------------------------- 10 export
md(r"""
## 10. Export the results

Packs `artepublica_results/` (the `results.json` with every number above, plus the figures)
into **`artepublica_results.zip`** and downloads it. It is small, a few megabytes.

**This zip is what to send back for the report.** If only one file can be sent, send
`results.json`.

The trained networks are not in the zip because they are large. They are in
`artepublica_checkpoints/` in the Files panel; set `DOWNLOAD_CHECKPOINTS = True` to download
them too. They load with `src/predict.py` from the repository.
""")

code(r"""
DOWNLOAD_CHECKPOINTS = False

save_results()
archive_path = shutil.make_archive("artepublica_results", "zip", OUTPUT_DIR)
print(f"wrote {archive_path} ({Path(archive_path).stat().st_size / 1e6:.1f} MB) containing:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {path.name}")

if IN_COLAB:
    colab_files.download(archive_path)
    if DOWNLOAD_CHECKPOINTS:
        for path in sorted(CHECKPOINT_DIR.glob("*.pt")):
            colab_files.download(str(path))
""")

md(r"""
## How to read these results with care

- **The validation score is optimistic.** Many photos were taken seconds apart from almost the
  same spot, so a validation photo often has a near-twin in the training set. Photos taken on
  another day, or from other angles, are the fair test.
- **Small differences are noise.** One validation photo is worth about 3 percentage points in
  a single run. Compare the ranges across seeds, not single numbers.
- **Only the final layer was trained.** Unfreezing the backbone (fine-tuning) on a GPU is the
  natural next experiment once there are more photos per sculpture.
""")


def main():
    notebook = new_notebook(cells=CELLS)
    notebook.metadata = {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    # Stable cell ids keep the generated file identical between runs.
    for number, cell in enumerate(notebook.cells):
        cell["id"] = f"cell-{number:02d}"
    out_path = Path(__file__).resolve().parent.parent / "notebooks" / "artepublica_colab.ipynb"
    out_path.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, out_path)
    print(f"wrote {out_path} ({len(notebook.cells)} cells)")


if __name__ == "__main__":
    main()
