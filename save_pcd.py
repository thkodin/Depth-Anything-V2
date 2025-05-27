import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from textwrap import dedent
from warnings import warn

import cv2
import numpy as np
from natsort import natsorted

VALID_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}
VALID_DEPTH_MAP_EXTENSIONS = {".npy"}
VALID_DEPTH_EXTENSIONS = VALID_IMG_EXTENSIONS | VALID_DEPTH_MAP_EXTENSIONS


@dataclass
class PcdImagePair:
    path_depth: Path
    path_color: Path | None = None
    is_raw_depth: bool = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save point cloud given a depth map (colored PCD if corresponding color image is provided).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path-depth",
        type=str,
        required=True,
        help="Path to depth image (.png/.jpg/.jpeg) or map (.npy format).",
        metavar="str",
    )
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs="+",
        required=True,
        help=(
            "Camera intrinsics [fx, fy, cx, cy] in that order. If one argument provided, fx and fy are both set equal"
            " to it and cx and cy are inferred from the image dimensions (width and height respectively). If 2 are"
            " provided, fx and fy are populated, and cx and cy are inferred from the image dimensions. If 3 are"
            " provided, cy is inferred from the image height. If 4 are provided, all values are populated."
        ),
        metavar="float",
    )
    parser.add_argument(
        "--path-color",
        type=str,
        help="Path to color image (RGB). If provided, the point cloud will be colored.",
        metavar="str",
    )
    parser.add_argument(
        "--dir-output",
        type=str,
        help=(
            "Path to output directory containing the point cloud data as PLY files. Defaults to 'pcd/YYYYMMDD_HHMMSS/'"
        ),
        metavar="str",
    )
    parser.add_argument(
        "--load-depth-map",
        action="store_true",
        default=False,
        help="Read raw depth maps instead of images (.npy format instead of .png/.jpg/.jpeg).",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help=(
            "Fixed depth scale factor. Where depth_image_px / depth_scale = depth_image_meters. IGNORED if"
            " --path-depth-scale is provided."
        ),
        metavar="float",
    )
    parser.add_argument(
        "--path-depth-scale",
        type=str,
        help=(
            "Path to file containing depth scale factors for each depth image as a JSON object representing a"
            " dictionary of the form {depth_image_filename: depth_scale_factor}. OVERRIDES --depth-scale."
        ),
        metavar="str",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.1,
        help="Match threshold for associating depth and normal images to color images based on filename similarity.",
        metavar="float",
    )
    parser.add_argument(
        "--recursive-search-input-directories",
        action="store_true",
        help="Search recursively through each input directory.",
    )
    parser.add_argument(
        "--pcd-image-center-as-principal-point",
        action="store_true",
        help="Use the image center as the principal point instead of the value provided in intrinsics.",
    )

    # Parse arguments.
    args = parser.parse_args()

    # Validate intrinsics.
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
            parser.error("--intrinsics must have between 1 and 4 values.")

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

def find_best_path_name_match(target: str, candidates: list[str], match_threshold: float = 0.1) -> str | None:
    """
    Find the best matching filename from candidates using fuzzy string matching.

    Args:
        target: The target filename to match against.
        candidates: List of candidate filenames.
        match_threshold: Minimum similarity ratio for a match.

    Returns:
        Best matching filename or None if no good match found.
    """
    if match_threshold < 0 or match_threshold > 1:
        raise ValueError(f"Invalid match threshold: {match_threshold}")

    # early return if no candidates.
    if not candidates:
        return None

    # Only match filename based on stem (no extensions).
    target_stem = Path(target).stem

    # Check for exact match first.
    for candidate in candidates:
        if Path(candidate).stem == target_stem:
            return candidate

    # If no exact match, proceed with fuzzy matching.
    best_ratio = 0
    best_match = None

    for candidate in candidates:
        candidate_stem = Path(candidate).stem
        ratio = SequenceMatcher(None, target_stem, candidate_stem).ratio()
        # if ratio == best_ratio and best_match is not None:
        #     warn(
        #         f"Filenames tied during match-based association: {Path(candidate).stem} (CURRENT) =="
        #         f" {Path(best_match).stem} (BEST). Will stick with BEST (older) one assuming filenames were sorted,"
        #         " such that this was the first best match."
        #     )
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate

    # Only return match if similarity is high enough.
    return best_match if best_ratio > match_threshold else None


def associate_images_for_pcd(
    dir_depth: Path, dir_color: Path | None = None, match_threshold: float = 0.5, load_depth_map: bool = False, recursive: bool = True
) -> list[PcdImagePair]:
    """
    Find any corresponding color images given depth images in the input directory.

    Args:
        dir_depth: Depth directory path.
        dir_color: Color directory path. May be None if only depth images are provided.
        match_threshold: Match threshold for associating depth images to color images based on filename similarity.
        load_depth_map: Whether to load raw depth maps instead of scaled depth images.
        recursive: Whether to search recursively through the input directories.

    Returns:
        List of PcdImagePair objects containing associated file paths.
    """
    if not dir_depth.is_dir():
        raise FileNotFoundError(f"Depth images directory not found: {dir_depth}")

    # Get list of depth images.
    paths_depth = natsorted(directory_tree_search(dir_depth, extensions=VALID_DEPTH_EXTENSIONS, recursive=recursive))
    if load_depth_map:
        # Keep just the .npy files.
        paths_depth = [f for f in paths_depth if f.suffix.lower() in VALID_DEPTH_MAP_EXTENSIONS]
    else:
        # Keep just the .png/.jpg/.jpeg files.
        paths_depth = [f for f in paths_depth if f.suffix.lower() in VALID_IMG_EXTENSIONS]

    # Get color files if directory exists.
    paths_color = []
    if dir_color is not None:
        if not dir_color.is_dir():
            raise FileNotFoundError(f"Color images directory not found: {dir_color}")
        paths_color = natsorted(directory_tree_search(dir_color, extensions=VALID_IMG_EXTENSIONS, recursive=recursive))

    # Associate files.
    image_sets = []
    for path_depth in paths_depth:
        path_color = None

        if paths_color:
            matched_path_color = find_best_path_name_match(
                path_depth.name, [f.name for f in paths_color], match_threshold=match_threshold
            )
            if matched_path_color:
                path_color = dir_color / matched_path_color

        image_sets.append(
            PcdImagePair(path_depth, path_color, is_raw_depth=path_depth.suffix.lower() in VALID_DEPTH_MAP_EXTENSIONS)
        )

    return image_sets


def create_point_cloud(
    depth: np.ndarray,
    intrinsics: list[float],
    color: np.ndarray = None,
    pcd_image_center_as_principal_point: bool = False,
) -> tuple:
    """
    Create a point cloud from depth map and camera intrinsics.

    Args:
        depth (np.ndarray): 2D depth map.
        intrinsics (list[float]): Camera intrinsics [fx, fy, cx, cy]. cx and cy may be None, in which case they are
            assumed to be the center of the depth image.
        color (np.ndarray, optional): RGB image for coloring points.
        pcd_image_center_as_principal_point (bool, optional): Whether to use the image center as the principal point instead
            of the value provided in intrinsics.

    Returns:
        tuple: Arrays of vertices and colors (if RGB provided).
    """

    # Create mesh grid of pixel coordinates.
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Convert to 3D points.
    fx, fy, cx, cy = intrinsics
    if cx is None or pcd_image_center_as_principal_point:
        cx = width / 2
    if cy is None or pcd_image_center_as_principal_point:
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


def validate_depth_image(depth: np.ndarray) -> np.ndarray:
    """
    Validate and process a depth image to ensure it's in the correct format.

    Args:
        depth (np.ndarray): The depth image to validate.

    Returns:
        np.ndarray: The validated and processed depth image.

    Raises:
        RuntimeError: If the depth image has invalid format.
    """
    if depth.ndim > 2:
        if all([dimsize > 1 for dimsize in depth.shape]):
            # Check if all channels are identical.
            first_channel = np.expand_dims(depth[..., 0], axis=-1)
            if not np.all(np.all(depth == first_channel, axis=-1)):
                raise RuntimeError(
                    f"Multi-channel grayscale image with non-identical channels at input. Shape was {depth.shape}."
                )
            depth = first_channel.squeeze()
        else:
            # We have a 3D grayscale image with a single element along the third axis, so we can squeeze it.
            depth = depth.squeeze()

    return depth


def load_depth_image(path: Path, depth_scale: float | dict, is_raw_depth: bool = False) -> np.ndarray:
    """
    Load and process a depth image from file.

    Args:
        path (Path): Path to the depth image.
        depth_scale (float | dict): Scale factor to convert depth values to metric units.
        is_raw_depth (bool, optional): Whether the file to load is a raw depth map (.npy).

    Returns:
        np.ndarray: The loaded and processed depth image.
    """
    if is_raw_depth:
        # Load raw depth map directly.
        depth = np.load(path.as_posix())
    else:
        # Load and scale depth image.
        depth = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
        if isinstance(depth_scale, dict):
            depth_scale = depth_scale[path.name]
        depth = depth / depth_scale

    depth = validate_depth_image(depth)
    return depth


def load_color_image(path: Path) -> np.ndarray:
    """
    Load a color image from file.

    Args:
        path (Path): Path to the color image.

    Returns:
        np.ndarray: The loaded and converted color image.
    """
    return cv2.cvtColor(cv2.imread(path.as_posix()), cv2.COLOR_BGR2RGB)


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


def main():
    # Parse command line arguments.
    args = parse_args()

    # Process arguments into local variables.
    path_depth = Path(args.path_depth)
    path_color = Path(args.path_color) if args.path_color else None
    dir_output = Path(args.dir_output) if args.dir_output else None
    load_depth_map = args.load_depth_map
    depth_scale = args.depth_scale
    path_depth_scale = Path(args.path_depth_scale) if args.path_depth_scale else None
    intrinsics = args.intrinsics
    match_threshold = args.match_threshold
    recursive_search_input_directories = args.recursive_search_input_directories
    pcd_image_center_as_principal_point = args.pcd_image_center_as_principal_point

    # Check if we're dealing with files or directories and that all of them are either files/directories.
    paths = [p for p in [path_depth, path_color] if p is not None]
    are_input_paths_files = all(p.is_file() for p in paths)
    are_input_paths_dirs = all(p.is_dir() for p in paths)

    if not (are_input_paths_files or are_input_paths_dirs):
        raise ValueError("Provided paths for depth and color images must be either all files or all directories.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Setup the output directory.
    if dir_output is not None:
        dir_output = dir_output / timestamp
    else:
        dir_output = Path(f"pcd/{timestamp}")

    run_config = {
        "timestamp": timestamp,
        "depth_path": path_depth.as_posix(),
        "color_path": path_color.as_posix() if path_color else None,
        "output_directory": dir_output.as_posix(),
        "load_depth_map": load_depth_map,
        "depth_scale": depth_scale,
        "depth_scale_file": path_depth_scale.as_posix() if path_depth_scale else None,
        "intrinsics": intrinsics,
        "match_threshold": match_threshold,
        "recursive_search_input_directories": recursive_search_input_directories,
        "pcd_image_center_as_principal_point": pcd_image_center_as_principal_point,
    }
    longest_key_length = max(len(k) for k in run_config.keys())
    arguments_text = dedent(
        f"""
        --------------------------------------------------------------------------------
        ARGUMENTS
        --------------------------------------------------------------------------------
        {"\n".join(f"{k:<{longest_key_length + 1}}: {v}" for k, v in run_config.items())}
        ================================================================================
        """
    )
    print(arguments_text)

    # Create output directory if it doesn't exist.
    dir_output.mkdir(parents=True, exist_ok=True)
    save_run_config(dir_output / "run_config.json", run_config)

    # Load the depth scale from the file if provided.
    if path_depth_scale:
        with open(path_depth_scale, "r") as f:
            depth_scale = json.load(f)

    # Get image sets based on whether we're dealing with files or directories.
    if are_input_paths_dirs:
        image_sets = associate_images_for_pcd(
            path_depth, dir_color=path_color, match_threshold=match_threshold, load_depth_map=load_depth_map, recursive=recursive_search_input_directories
        )
    elif are_input_paths_files:
        # Single file case - create a single ImageSet.
        image_sets = [
            PcdImagePair(path_depth, path_color, is_raw_depth=path_depth.suffix.lower() in VALID_DEPTH_MAP_EXTENSIONS)
        ]
    else:
        raise ValueError("Invalid input paths.")

    print(f"Found {len(image_sets)} images to process.")

    # Process each image set.
    for i, image_set in enumerate(image_sets):
        print(f"Processing image set {i+1}/{len(image_sets)}: {image_set.path_depth.name}")

        # Load depth image.
        depth = load_depth_image(
            path=image_set.path_depth, depth_scale=depth_scale, is_raw_depth=image_set.is_raw_depth
        )

        # Read color image if available.
        color = None
        if image_set.path_color:
            color = load_color_image(image_set.path_color)

        # Create and save point cloud.
        points, colors = create_point_cloud(
            depth, intrinsics, color, pcd_image_center_as_principal_point=pcd_image_center_as_principal_point
        )
        output_path = dir_output / f"{image_set.path_depth.stem}_pcd.ply"
        save_point_cloud(points, colors, output_path)


if __name__ == "__main__":
    main()
