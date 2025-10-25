import cv2
import torch
from dataset.transform import NormalizeImage, PrepareForNet, Resize
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class KnottyCaptcha(Dataset):
    """
    Dataloader for Knotty CAPTCHA dataset. The actual dataloading pipeline can become a lot more complex depending on
    the Blender configuration, since the depth scale can change with the Camera's far clip distance as well as changes
    in depth storage formats, etc.

    This dataset is generated from Blender, usually with some far clip distance on the camera, beyond which the camera
    does not render any objects. The depth renders are scaled according to this value in the range 0-65535 (16-bit PNG
    format).  Thus, to recover the metric depth, we need to scale all images DOWN by 65535/far_clip_distance =
    depth_scale_factor.
    """

    def __init__(self, filelist_path, mode, size=(518, 518), depth_scale_factor=7710.0):
        if mode not in ["train", "val", "test"]:
            raise ValueError("Mode must be one of: train, val, test")

        self.mode = mode
        self.size = size
        self.depth_scale_factor = depth_scale_factor

        with open(filelist_path, "r") as f:
            self.filelist = f.read().splitlines()

        net_w, net_h = size
        self.transform = Compose(
            [
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=True if mode == "train" else False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def __getitem__(self, item):
        img_path = self.filelist[item].split(" ")[0]
        depth_path = self.filelist[item].split(" ")[1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype("float32")

        sample = self.transform({"image": image, "depth": depth})

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        # Magic number explanations in module docstring.
        sample["depth"] = sample["depth"] / self.depth_scale_factor
        sample["valid_mask"] = sample["depth"] > 0
        sample["image_path"] = self.filelist[item].split(" ")[0]

        return sample

    def __len__(self):
        return len(self.filelist)
