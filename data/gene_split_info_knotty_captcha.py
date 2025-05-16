"""
Make the .txt pairwise format required by Depth-Anything-V2. These split files are located at: metric_depth/dataset/splits/{dataset_name}/{split_name}.txt

WARNING: The authors use space-separated text files, but this is not a robust method as spaces in file paths will ruin
this approach. It would be much preferable to utilize a JSON format with explicit indication of color/depth paths, and
possibly even define a relative data root within the project directory to minimize path lengths.

Each line contains two space-separated strings, representing paths to the RGB and depth image in that order. For example:

/path/to/rgb/image1.png /path/to/depth/image1.png
/path/to/rgb/image2.png /path/to/depth/image2.png
...

For the knotty_captcha dataset, the RGB and depth image files have mostly the same stem, but are suffixed with _color or
_depth - their extensions may not necessarily be the same as well. We don't care about the normal images for DA-V2. The
data structure should look like this:

knotty_captcha/
└── train/
    ├── color/
    │   ├── image1_color.png
    │   ├── image2_color.png
    └── depth/
        ├── image1_depth.png
        └── image2_depth.png
"""

from pathlib import Path

from natsort import natsorted
from tqdm import tqdm


def main():
    DATASET_NAME = "knotty_captcha"
    ROOT_REPO = Path(__file__).parents[1]
    DIR_DATASET = ROOT_REPO / "data" / DATASET_NAME
    DIR_OUTPUT = ROOT_REPO / "metric_depth" / "dataset" / "splits" / DATASET_NAME
    SPLITS = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}
    DIR_COLOR_IMAGES_IN_SPLIT_DIR = "color"
    DIR_DEPTH_IMAGES_IN_SPLIT_DIR = "depth"
    COLOR_IMAGE_SUFFIX = "_color"
    DEPTH_IMAGE_SUFFIX = "_depth"
    VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

    for split_name, split_file in SPLITS.items():
        # Recursively look for corresponding directory with {split_name} in the dataset directory.
        dir_split = natsorted(DIR_DATASET.glob(f"**/{split_name}"))
        if len(dir_split) == 0:
            raise ValueError(f"Directory with split name {split_name} not found in {DIR_DATASET}.")
        elif len(dir_split) > 1:
            raise ValueError(
                f"Multiple directories with split name {split_name} found in {DIR_DATASET}. Please ensure only one such"
                " directory exists as it's otherwise ambiguous which one to use."
            )
        dir_split = dir_split[0]

        # Now we're within the split directory. Now, we need to get the RGB and depth images as associated pairs.
        dir_color_images = dir_split / DIR_COLOR_IMAGES_IN_SPLIT_DIR
        dir_depth_images = dir_split / DIR_DEPTH_IMAGES_IN_SPLIT_DIR

        # We'll store the RGB and depth image paths in a list.
        paths_color_images = natsorted(
            item for item in dir_color_images.iterdir() if item.is_file() and item.suffix in VALID_IMAGE_EXTENSIONS
        )
        list_rgb_depth_pairs = []
        for path_color in tqdm(paths_color_images, desc=f"Processing {split_name} split"):
            # Since depth image suffix might be different than RGB, we run a wildcard match.
            path_associated_depth = [
                path
                for path in natsorted(
                    dir_depth_images.glob(f"{path_color.stem.replace(COLOR_IMAGE_SUFFIX, DEPTH_IMAGE_SUFFIX)}*")
                )
                if path.suffix in VALID_IMAGE_EXTENSIONS
            ]
            if len(path_associated_depth) == 0:
                raise ValueError(f"No associated depth image found for {path_color}.")
            elif len(path_associated_depth) > 1:
                raise ValueError(f"Multiple associated depth images found for {path_color}.")
            path_associated_depth = path_associated_depth[0]
            list_rgb_depth_pairs.append(f"{path_color.as_posix()} {path_associated_depth.as_posix()}")

        # Write the list of RGB and depth image paths to the output file.
        split_path = DIR_OUTPUT / split_file
        with open(split_path, "w") as f:
            for line in list_rgb_depth_pairs:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
