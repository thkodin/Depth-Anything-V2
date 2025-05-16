import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def image_tree_search(path: Path):
    """Iteratively search for all files with the extension .jpg or .png in the directory tree."""
    filenames = []
    stack = [path]

    while stack:
        item = stack.pop()
        if item.is_file() and item.suffix in [".jpg", ".png"]:
            filenames.append(item)
        elif item.is_dir():
            for entry in item.iterdir():
                if entry.is_file() and entry.suffix in [".jpg", ".png"]:
                    filenames.append(entry)
                elif entry.is_dir():
                    stack.append(entry)
        else:
            raise RuntimeError(f"is not file or directory: {item}")

    return filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Anything V2")

    parser.add_argument("--img-path", type=str)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--outdir", type=str, default="./vis_depth")

    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])

    parser.add_argument("--pred-only", dest="pred_only", action="store_true", help="only display the prediction")
    parser.add_argument("--grayscale", dest="grayscale", action="store_true", help="do not apply colorful palette")

    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(
        torch.load(Path("checkpoints") / f"depth_anything_v2_{args.encoder}.pth", map_location="cpu")
    )
    depth_anything = depth_anything.to(DEVICE).eval()

    img_path = Path(args.img_path)
    if img_path.is_file():
        if img_path.suffix == ".txt":
            filenames = [Path(line.strip()) for line in img_path.read_text().splitlines()]
        else:
            filenames = [img_path]
    else:
        filenames = image_tree_search(img_path)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    depth_scales = {}
    for k, filename in enumerate(filenames):
        print(f"Progress {k+1}/{len(filenames)}: {filename}")

        original_image = cv2.imread(filename.as_posix())
        original_height, original_width = original_image.shape[:2]
        output_path = outdir / f"{filename.stem}.png"

        # Get the raw, metric depth map.
        depth = depth_anything.infer_image(original_image, args.input_size)
        # Resize to original image size.
        depth = cv2.resize(depth, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        print(f"RAW DEPTH (METERS): {depth.min():.2f} min, {depth.max():.2f} max")
        # Get the depth scale.
        depth_scale = 65535.0 / depth.max()
        depth_scales[output_path.name] = depth_scale
        # Scale the depth map to 0-65535 uint16.
        depth = depth * depth_scale
        # Ensure type is uint16.
        depth = depth.astype(np.uint16)

        if not args.grayscale:
            # This is a 3D color image, not suitable for PCD generation. First, squeeze the uint16 range to 0-255 uint8.
            depth = (depth / 65535.0 * 255.0).astype(np.uint8)
            # Then apply the colormap.
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        if args.pred_only:
            # Write the 16-bit (if grayscale) or 8-bit (if colormapped) depth map.
            cv2.imwrite(output_path.as_posix(), depth)
        else:
            if depth.ndim == 2:
                # Convert the depth image to a 3-channel color-like image representing the depth. This would be the case
                # if args.grayscale is True. Again, we squeeze the uint16 range to 0-255 uint8, and repeat the depth
                # channel 3 times to make it a 3-channel image.
                depth = (depth / 65535.0 * 255.0).astype(np.uint8)
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

            split_region = np.ones((original_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([original_image, split_region, depth])
            cv2.imwrite(output_path.as_posix(), combined_result)

    with open(outdir / f"depth_scales.json", "w") as f:
        json.dump(depth_scales, f)
