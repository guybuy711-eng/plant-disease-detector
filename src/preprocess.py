import os
from PIL import Image
from pathlib import Path


def clean_dataset(data_dir):
    removed_imgs = 0
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for path in Path(data_dir).rglob(ext):
            try:
                img = Image.open(path)
                img.verify()
            except Exception as e:
                os.remove(path)
                removed_imgs += 1
                print(f"Removed img because of: {e}")
    return removed_imgs

if __name__ == "__main__":
    removed = clean_dataset(data_dir="data/raw/plantvillage dataset/color")
    print(f"Total removed: {removed}")
