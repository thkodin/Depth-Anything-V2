"""
Run a downloaded DepthAnythingV2 metric depth checkpoint. Does not support auto-downloading - please download the
checkpoint manually before running this script.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import cv2
import numpy as np
import torch
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2

TRAIN_TIME_KEYS = ["model", "optimizer", "epoch", "previous_best"]
VALID_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Depth Anything V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path-color",
        type=str,
        help="Path to the color image. Can be a file or a directory.",
        required=True,
        metavar="str",
    )
    parser.add_argument(
        "--dir-output",
        type=str,
        help="Output directory. Defaults to 'inference/{model_name}/YYYYMMDD_HHMMSS/'",
        metavar="str",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vitl",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Encoder to use.",
        metavar="str",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help=(
            "Input size for the model (images resized to this size). If a non-positive value is provided, the original"
            " image height is used (height is important for depth calculations)."
        ),
        metavar="int",
    )
    parser.add_argument(
        "--recursive-search-input-directories",
        action="store_true",
        help="Search for input files in subdirectories of the provided path.",
    )
    parser.add_argument(
        "--save-pcds",
        action="store_true",
        help="Save point clouds.",
    )
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs="*",
        help=(
            "Camera intrinsics [fx, fy, cx, cy] in that order. Required for point cloud generation. If one argument"
            " provided, fx and fy are both set equal to it and cx and cy are inferred from the image dimensions (width"
            " and height respectively). If 2 are provided, fx and fy are populated, and cx and cy are inferred from the"
            " image dimensions. If 3 are provided, cy is inferred from the image height. If 4 are provided, all values"
            " are populated."
        ),
        default=[470.4, 470.4],
        metavar="float",
    )

    args = parser.parse_args()

    match len(args.intrinsics):
        case 1:
            args.intrinsics.extend([args.intrinsics[0], None, None])
        case 2:
            args.intrinsics.extend([None, None])
        case 3:
            args.intrinsics.append(None)
        case 4:
            pass
        case _:
            raise ValueError("--intrinsics must have between 1 and 4 values.")

    return args


def directory_tree_search(path: Path, extensions: set[str] | None = None, recursive: bool = True):
    """
    Iteratively search for all files with the specified extensions in the directory tree.

    Args:
        path: The root path to start searching from.
        extensions: Set of file extensions to filter by. If None, uses VALID_IMG_EXTENSIONS.
        recursive: Whether to search recursively through subdirectories.
    """
    # Use default extensions if none provided.
    if extensions is None:
        extensions = set()

    filenames = []
    stack = [path]

    while stack:
        item = stack.pop()
        if item.is_file():
            # Add file if no extensions specified or if extension matches.
            if not extensions or item.suffix.lower() in extensions:
                filenames.append(item)
        elif item.is_dir():
            for entry in item.iterdir():
                if entry.is_file():
                    # Add file if no extensions specified or if extension matches.
                    if not extensions or entry.suffix.lower() in extensions:
                        filenames.append(entry)
                elif entry.is_dir() and recursive:
                    stack.append(entry)
        else:
            raise RuntimeError(f"is not file or directory: {item}")

    return filenames


def save_run_config(save_path: Path, config_dict: dict) -> None:
    """
    Save the run configuration to a JSON file.

    Args:
        save_path (Path): Path to save the config file to. Must be either .json or .txt.
        config_dict (dict): Dictionary containing the processed configuration parameters.
    """
    if not save_path.suffix in [".json", ".txt"]:
        raise ValueError("Run config file must have .json or .txt extension.")
    with open(save_path, "w") as f:
        if save_path.suffix == ".json":
            json.dump(config_dict, f, indent=4)
        else:
            longest_key_length = max(len(k) for k in config_dict.keys())
            f.writelines(f"{k:<{longest_key_length + 1}}: {v}\n" for k, v in config_dict.items())


def load_color_image(path: Path) -> np.ndarray:
    """
    Load a color image from file. Depth-Anything V2 expects an RGB image at input, but its inference method
    infer_image() automatically converts the image to RGB from BGR. So we mustn't do that here.

    Args:
        path (Path): Path to the color image.

    Returns:
        np.ndarray: The loaded and converted color image.
    """
    return cv2.imread(path.as_posix())


def create_point_cloud(
    depth: np.ndarray,
    intrinsics: list[float],
    color: np.ndarray = None,
    image_center_as_principal_point: bool = False,
) -> tuple:
    """
    Create a point cloud from depth map and camera intrinsics.

    Args:
        depth (np.ndarray): 2D depth map.
        intrinsics (list[float]): Camera intrinsics [fx, fy, cx, cy]. cx and cy may be None, in which case they are
            assumed to be the center of the depth image.
        color (np.ndarray, optional): RGB image for coloring points.
        image_center_as_principal_point (bool, optional): Whether to use the image center as the principal point instead
            of the value provided in intrinsics.

    Returns:
        tuple: Arrays of vertices and colors (if RGB provided).
    """
    # Create mesh grid of pixel coordinates.
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Convert to 3D points.
    fx, fy, cx, cy = intrinsics
    if cx is None or image_center_as_principal_point:
        cx = width / 2
    if cy is None or image_center_as_principal_point:
        cy = height / 2

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # Stack coordinates.
    points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    mask = points[:, 2] > 0

    # Filter out the points with no depth.
    points = points[mask]
    colors = None
    if color is not None:
        colors = color.reshape(-1, 3)
        colors = colors[mask]

    return points, colors


def save_point_cloud(points: np.ndarray, colors: np.ndarray, save_path: Path) -> None:
    """
    Save point cloud to PLY file.

    Args:
        points (np.ndarray): Point coordinates.
        colors (np.ndarray): Point colors.
        filepath (Path): Output filepath.
    """
    # Prepare data for writing.
    if colors is not None:
        data = np.hstack([points, colors])
        fmt = "%.6f %.6f %.6f %d %d %d"
    else:
        data = points
        fmt = "%.6f %.6f %.6f"

    # Write header and data in a single operation.
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if colors is not None:
        header.extend(["property uchar red", "property uchar green", "property uchar blue"])

    header.append("end_header")
    header = "\n".join(header)

    # Save with header.
    np.savetxt(save_path, data, fmt=fmt, header=header, comments="")


if __name__ == "__main__":
    args = parse_args()

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    # Process arguments into local variables.
    path_color = Path(args.path_color)
    dir_output = Path(args.dir_output) if args.dir_output else None
    encoder = args.encoder
    input_size = args.input_size
    recursive_search_input_directories = args.recursive_search_input_directories
    save_pcds = args.save_pcds
    intrinsics = args.intrinsics

    # Validate intrinsics.
    match len(intrinsics):
        case 1:
            intrinsics.extend([intrinsics[0], None, None])
        case 2:
            intrinsics.extend([None, None])
        case 3:
            intrinsics.append(None)
        case 4:
            pass
        case _:
            raise ValueError("--intrinsics must have between 1 and 4 values.")

    # Check if we're dealing with files or directories.
    if not (path_color.is_file() or path_color.is_dir()):
        raise ValueError("Provided path for color images must be either a file or a directory.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dir_output is not None:
        dir_output = dir_output / timestamp
    else:
        dir_output = Path("inference") / encoder / timestamp

    run_config = {
        "timestamp": timestamp,
        "path_color": path_color.as_posix(),
        "dir_output": dir_output.as_posix(),
        "encoder": encoder,
        "input_size": input_size,
        "intrinsics": intrinsics,
        "recursive_search_input_directories": recursive_search_input_directories,
        "save_pcds": save_pcds,
    }
    longest_key_length = max(len(k) for k in run_config.keys())
    argument_lines = [f"{k:<{longest_key_length + 1}}: {v}" for k, v in run_config.items()]
    # Strip each line of leading spaces.
    argument_lines = "\n".join([line.lstrip() for line in argument_lines])
    argument_lines = f"""
        {"-" * 120}
        ARGUMENTS
        {"-" * 120}
        {argument_lines}
        {"=" * 120}
    """
    # Strip each line of leading spaces.
    argument_lines = "\n".join([line.lstrip() for line in argument_lines.split("\n")])
    print(argument_lines)

    dir_depth = dir_output / "depth"
    dir_pcd = dir_output / "pcd"
    path_depth_scale = dir_output / "depth_scales.json"

    # Create output directory.
    dir_output.mkdir(parents=True, exist_ok=True)
    dir_depth.mkdir(parents=True, exist_ok=True)
    if save_pcds:
        dir_pcd.mkdir(parents=True, exist_ok=True)

    save_run_config(dir_output / "run_config.json", run_config)

    model = DepthAnythingV2(**model_configs[encoder])
    state_dict: dict = torch.load(f"checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu")
    if all(key in state_dict for key in TRAIN_TIME_KEYS):
        if "model" in state_dict:
            state_dict = state_dict["model"]
        else:
            raise ValueError(f"Model key not found in trained model state dictionary: {state_dict.keys()}")
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()

    # Get list of images to process.
    if path_color.is_file():
        paths_color = [path_color]
    else:
        paths_color = directory_tree_search(
            path_color, VALID_IMG_EXTENSIONS, recursive=recursive_search_input_directories
        )

    print(f"Found {len(paths_color)} images to process.")

    depth_scales = {}

    for k, path_color in tqdm(enumerate(paths_color), total=len(paths_color), desc="Processing Images"):
        original_image = load_color_image(path_color)
        original_height, original_width = original_image.shape[:2]
        path_output_depth_image = dir_depth / f"{path_color.stem}_depth_scaled.png"
        path_output_pcd = dir_pcd / f"{path_color.stem}.ply"

        # Get the raw depth map.
        depth = model.infer_image(original_image, input_size)
        # Resize to original image size.
        depth = cv2.resize(depth, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        # print(f"RAW DEPTH (METERS): {depth.min():.2f} min, {depth.max():.2f} max")

        # Get the depth scale.
        depth_scale = 65535.0 / depth.max()
        depth_scales[path_output_depth_image.name] = depth_scale
        # Scale the depth map to 0-65535 uint16 and write to disk.
        cv2.imwrite(path_output_depth_image.as_posix(), scaled_depth := (depth * depth_scale).astype(np.uint16))

        # Save point cloud if requested.
        if save_pcds:
            # Convert depth back to meters for point cloud generation.
            depth_meters = depth / depth_scale
            points, colors = create_point_cloud(depth_meters, intrinsics, original_image)
            save_point_cloud(points, colors, dir_pcd / f"{path_color.stem}.ply")

    with open(path_depth_scale, "w") as f:
        json.dump(depth_scales, f)
