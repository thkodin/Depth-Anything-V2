"""
Split a dataset of blender renders into train/val/test splits for fine-tuning models.

The split knotty_captcha dataset will look like this (containing color, depth, and normal images):

knotty_captcha/
└── train/
    ├── color/
    │   ├── image1_color.png
    │   └── image2_color.png
    ├── depth/
    │   ├── image1_depth.png
    │   └── image2_depth.png
    └── normal/
        ├── image1_normal.png
        └── image2_normal.png
"""

import random
import shutil
from pathlib import Path

from natsort import natsorted

# The root data for images is the directory containing the images.
REPO_ROOT = Path(__file__).parents[1]
DIR_DATA_ROOT = REPO_ROOT / "data" / "knotty_captcha"
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

COLOR_IMAGE_SUFFIX = "_color"
DEPTH_IMAGE_SUFFIX = "_depth"
NORMAL_IMAGE_SUFFIX = "_normal"

FP_ERROR_TOLERANCE = 1e-6

# Split settings.
SPLIT_PROPORTIONS = (0.8, 0.1, 0.1)
SHUFFLE = True
SEED = 42  # set to None to disable


def split_dataset(
    base_dir: Path,
    split_proportions: tuple[float, float, float],
    shuffle: bool = True,
    seed: int = None,
):
    """Split the dataset into train/val/test splits.

    Args:
        base_dir: The base directory containing the color, depth, and normal image folders.
        split_proportions: The proportions of the dataset to split into train/val/test.
        shuffle: Whether to shuffle the dataset.
        seed: The seed to use for shuffling the dataset. If None, no shuffling is done.
    """
    if seed is not None:
        random.seed(seed)

    def move_images(indices: list[int], split: str):
        split_dir = base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for idx in indices:
            (split_dir / "color").mkdir(parents=True, exist_ok=True)
            (split_dir / "depth").mkdir(parents=True, exist_ok=True)
            shutil.move(color_images[idx], split_dir / "color" / color_images[idx].name)
            shutil.move(depth_images[idx], split_dir / "depth" / depth_images[idx].name)
            if normal_images:
                (split_dir / "normal").mkdir(parents=True, exist_ok=True)
                shutil.move(normal_images[idx], split_dir / "normal" / normal_images[idx].name)

    color_dir = base_dir / "color"
    depth_dir = base_dir / "depth"
    normal_dir = base_dir / "normal"

    if not color_dir.exists() or not any(color_dir.glob(f"*{ext}" for ext in VALID_IMAGE_EXTENSIONS)):
        raise ValueError("Color directory does not exist or does not contain any valid images.")
    if not depth_dir.exists() or not any(depth_dir.glob(f"*{ext}" for ext in VALID_IMAGE_EXTENSIONS)):
        raise ValueError("Depth directory does not exist or does not contain any valid images.")

    color_images = natsorted(
        item for item in color_dir.iterdir() if item.is_file() and item.suffix in VALID_IMAGE_EXTENSIONS
    )
    depth_images = natsorted(
        item for item in depth_dir.iterdir() if item.is_file() and item.suffix in VALID_IMAGE_EXTENSIONS
    )
    normal_images = natsorted(
        item for item in normal_dir.iterdir() if item.is_file() and item.suffix in VALID_IMAGE_EXTENSIONS
    )

    # Validate the data.
    if len(color_images) != len(depth_images):
        raise ValueError("Color and depth images count mismatch.")
    if normal_images and len(color_images) != len(normal_images):
        raise ValueError("Color and normal images count mismatch.")

    # Ensure that the all images are paired correctly based on the common part of their filestem.
    for color_image, depth_image, normal_image in zip(color_images, depth_images, normal_images):
        if color_image.stem.removesuffix(COLOR_IMAGE_SUFFIX) != depth_image.stem.removesuffix(DEPTH_IMAGE_SUFFIX):
            raise ValueError("Color and depth images are not paired correctly.")
        if normal_images and color_image.stem.removesuffix(COLOR_IMAGE_SUFFIX) != normal_image.stem.removesuffix(
            NORMAL_IMAGE_SUFFIX
        ):
            raise ValueError("Color and normal images are not paired correctly.")

    total_images = len(color_images)
    indices = list(range(total_images))

    if shuffle:
        random.shuffle(indices)

    train_split = int(split_proportions[0] * total_images)
    val_split = train_split + int(split_proportions[1] * total_images)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    move_images(train_indices, "train")
    move_images(val_indices, "val")
    move_images(test_indices, "test")

    shutil.rmtree(color_dir)  # Remove the original color directory.
    shutil.rmtree(depth_dir)  # Remove the original depth directory.
    if normal_dir.exists():
        shutil.rmtree(normal_dir)  # Remove the original normal directory if it exists.


def main():
    # This is the directory containing the color, depth, and optionally normal image folders.
    if not abs(1.0 - sum(SPLIT_PROPORTIONS)) < FP_ERROR_TOLERANCE:
        raise ValueError(f"Split proportions must sum to 1 within float-point error +-{FP_ERROR_TOLERANCE}.")

    split_dataset(
        base_dir=DIR_DATA_ROOT,
        split_proportions=SPLIT_PROPORTIONS,
        shuffle=SHUFFLE,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
