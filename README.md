# artepublica-AI

Image recognition built on a pretrained network: an ImageNet-pretrained ResNet-50 whose
backbone is frozen, with a new classification head trained on this project's images.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Images live in `data/raw/`, one subfolder per class:

```
data/raw/
  class_a/  img1.jpg  img2.jpg ...
  class_b/  ...
```

They are committed to the repository, so cloning it is enough to get them. To re-import from
the source Google Drive folder (shared by public link; needs a network that can reach Drive):

```bash
python scripts/download_data.py "<drive-folder-link>"
```

## Train

```bash
python src/train.py --data data/raw --epochs 10
```

Holds out 20% of each class for validation and saves the best checkpoint, with the class
names, to `models/classifier.pt` (not committed).

## Recognize new images

```bash
python src/predict.py path/to/image.jpg
```

Prints the top 3 classes with their confidence.
