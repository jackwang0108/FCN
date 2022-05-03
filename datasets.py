# Standard Library
from typing import *
from pathlib import Path

# Third-Party Library
import numpy as np
import PIL.Image as Image
from colorama import Fore, Style, init

# Torch Library
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# My Library
from helper import DatasetPath, ColorClsNameLookup
from helper import get_image, visualize

init(autoreset=True)

ImageType = TypeVar("ImageType", np.ndarray, torch.Tensor, Image.Image)
LabelType = TypeVar("LabelType", int, np.ndarray, torch.Tensor)


class PascalVOC2012Datasets(data.Dataset):

    def __init__(self, split: str):
        super(PascalVOC2012Datasets, self).__init__()

        assert split in ["train", "val"], f""
        self.split = split

        if self.split == "train":
            kw = "train"
        else:
            kw = "val"
        pascal_folder = DatasetPath.PascalVoc2012(split=self.split)
        self.image = [pascal_folder.image_folder.joinpath(f"{i}.jpg") for i in getattr(pascal_folder, f"{kw}_split")]
        self.target = [pascal_folder.target_folder.joinpath(f"{i}.png") for i in getattr(pascal_folder, f"{kw}_split")]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def set_transform(self, trans: Optional[transforms.Compose] = None):
        normalize_tans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if trans is None:
            # use the mean and std of imagenet
            self.transform = normalize_tans
        else:
            self.transform = transforms.Compose([
                trans,
                normalize_tans
            ])


if __name__ == "__main__":
    import pprint

    pv2012 = PascalVOC2012Datasets("train")
    print(all(i.exists() for i in pv2012.image))
    print(all(i.exists() for i in pv2012.target))
