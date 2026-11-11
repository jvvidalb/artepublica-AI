# artepublica-AI

Image recognition built on a pretrained network: an ImageNet-pretrained ResNet-50 whose
backbone is frozen, with a new classification head trained on this project's images.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Photos of four public sculptures in Rio de Janeiro, imported from Google Drive and sorted by
hand into one folder per sculpture:

| Folder in `data/raw/` | Images |
| --- | --- |
| `escultura_para_o_rio` | 64 |
| `o_passante` | 37 |
| `retangulo_vazado` | 38 |
| `sem_titulo_amilcar_de_castro` | 31 |

`data/other/` holds files from the same Drive folder that are not used for training: newspaper
clippings, plaque photos, street views without a sculpture, a collage and cover artwork.
Exact duplicate files were dropped.

Everything is committed, so cloning the repository is enough to get the images. To fetch the
Drive folder again (it must be shared by public link, and the network must reach Drive):

```bash
python scripts/download_data.py "<drive-folder-link>" --out drive_download
```

The Drive folder is flat, so newly downloaded photos have to be moved into the right
`data/raw/<sculpture>/` folder by hand. To add a new sculpture, create a new folder.

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
