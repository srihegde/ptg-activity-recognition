'''

TODO:
* Initialize with requirements: inputs and outputs
* Implement __getitem__() method
* Cleaning and formatting
* Add comments + docs

'''

import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from typing import List, Union, Tuple, Any


class H2OFrameDataset(torch.utils.data.Dataset):
    """
    A dataset class to load labels per frame

    Args:
    root_dir (string): Directory with all the images.
    mode (string): Denotes the split of data to be loaded (train (default) | test | val)
    transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir: str, mode: str = "train", transform = None):
        super(H2OFrameDataset, self).__init__()

        self.root_dir = root_dir
        self.frame_list = self._get_frame_list(mode)
        self.transform = transform


    def _load_image(self, directory: str, idx: int) -> Image.Image:
        return Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert('RGB')


    def __len__(self) -> int:
        return len(self.video_list)


    def __getitem__(self, idx: int):
        # return sample
        pass