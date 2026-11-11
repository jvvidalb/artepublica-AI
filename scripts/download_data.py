"""Download a publicly shared Google Drive folder and check that every file is a readable image.

Usage: python scripts/download_data.py <drive-folder-link> [--out data/raw]

Subfolders are kept as they are in Drive; training expects one subfolder per class.
Needs a network that can reach Google Drive.
"""

import argparse
from pathlib import Path

import gdown
from PIL import Image


def check_images(root):
    """Return (image_count, total_bytes, unreadable_paths) for all files under root."""
    count = total_bytes = 0
    unreadable = []
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        total_bytes += path.stat().st_size
        try:
            with Image.open(path) as image:
                image.verify()
            count += 1
        except Exception:
            unreadable.append(path)
    return count, total_bytes, unreadable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="link to a Drive folder shared as 'anyone with the link'")
    parser.add_argument("--out", default="data/raw", help="destination folder")
    args = parser.parse_args()

    downloaded = gdown.download_folder(url=args.url, output=args.out, resume=True)
    if not downloaded:
        raise SystemExit("Nothing was downloaded. Check that the folder is shared by public link.")

    count, total_bytes, unreadable = check_images(args.out)
    print(f"{count} readable images, {total_bytes / 1e6:.1f} MB in {args.out}")
    if unreadable:
        print(f"{len(unreadable)} files are not readable images:")
        for path in unreadable:
            print(f"  {path}")


if __name__ == "__main__":
    main()
