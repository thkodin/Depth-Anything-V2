"""
Dataloader for Knotty CAPTCHA dataset. The actual dataloading pipeline can become a lot more complex depending on the
Blender configuration, since the depth scale can change with the Camera's far clip distance as well as changes in depth
storage formats, etc.

This dataloader is currently hardcoded for a far clip distance of 8.5 meters and depth images in 16-bit PNG format,
which results in a depth scale factor of 65535/8.5 = 7710.0.
"""

import cv2
import torch
from dataset.transform import NormalizeImage, PrepareForNet, Resize
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class KnottyCaptcha(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        if mode not in ["train", "val", "test"]:
            raise ValueError("Mode must be one of: train, val, test")

        self.mode = mode
        self.size = size

        with open(filelist_path, "r") as f:
            self.filelist = f.read().splitlines()

        net_w, net_h = size
        self.transform = Compose([
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
        ])

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
        sample["depth"] = sample["depth"] / 7710.0
        sample["valid_mask"] = sample["depth"] > 0
        sample["image_path"] = self.filelist[item].split(" ")[0]

        return sample

    def __len__(self):
        return len(self.filelist)
