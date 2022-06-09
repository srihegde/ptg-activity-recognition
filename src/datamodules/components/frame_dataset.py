"""

TODO:
* Add comments + docs

"""

import os
import os.path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class H2OFrameDataset(torch.utils.data.Dataset):
    """A dataset class to load labels per frame.

    Args:
    root_dir (string): Directory with all the images.
    mode (string): Denotes the split of data to be loaded (train (default) | test | val)
    transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, root_dir: str, mode: str = "train", transform=None):
        super(H2OFrameDataset, self).__init__()

        self.root_dir = root_dir
        self.frame_list = self._get_frame_list(mode)
        self.transform = transform

    def _load_image(self, frame_file: str) -> Image.Image:
        frame = Image.open(frame_file).convert("RGB")
        if self.transform is not None:
            frame = self.transform(frame)
        return frame

    def _load_hand_pose(self, annotation_file: str) -> Tuple[List[float], List[float]]:
        with open(annotation_file) as f:
            lines = f.readlines()[0].strip().split()
            num_kpts = 21

            if int(float(lines[0])) == 1:
                lefth = [float(pos) for pos in lines[1 : (num_kpts * 3 + 1)]]
            else:
                lefth = np.zeros(num_kpts * 3).tolist()
            if int(float(lines[num_kpts * 3 + 1])) == 1:
                righth = [float(pos) for pos in lines[(num_kpts * 3 + 2) :]]
            else:
                righth = np.zeros(num_kpts * 3).tolist()

        return lefth, righth

    def _load_obj_data(self, annotation_file: str) -> Tuple[int, List[float]]:
        with open(annotation_file) as f:
            lines = f.readlines()[0].strip().split()
            obj_class = int(float(lines[0]))
            obj_pose = [float(c) for c in lines[1:]]

        return obj_class, obj_pose

    def _load_verb(self, annotation_file: str) -> int:
        with open(annotation_file) as f:
            lines = f.readlines()[0].strip().split()
            verb = int(float(lines[0]))

        return verb

    def _get_frame_list(self, mode: str) -> List[str]:
        data_file = os.path.join(self.root_dir, f"label_split/pose_{mode}.txt")
        frame_list: List[str]
        with open(data_file) as f:
            lines = f.readlines()
            frame_list = [os.path.join(self.root_dir, line.strip()) for line in lines]

        return frame_list

    def __len__(self) -> int:
        return len(self.frame_list)

    def __getitem__(self, idx: int):
        """For frame with id idx, loads frame with corresponding.

        labels - hand pose, object pose, object label, verb

        Args:
            idx: Frame sample index.
        Returns:
            A tuple of (frame, label). Label is a dictionary consisting
            of hand pose, object pose, object label, verb labels
        """
        frm = self._load_image(self.frame_list[idx])
        v_path = self.frame_list[idx].split("/")
        path_list = "/".join(v_path[:-2])
        fname = v_path[-1].split(".")[0]

        l_hand, r_hand = self._load_hand_pose(
            os.path.join(path_list, "hand_pose", f"{fname}.txt")
        )
        obj_label, obj_pose = self._load_obj_data(
            os.path.join(path_list, "obj_pose", f"{fname}.txt")
        )
        verb = self._load_verb(os.path.join(path_list, "verb_label", f"{fname}.txt"))

        return {
            "frm": frm,
            "l_hand": l_hand,
            "r_hand": r_hand,
            "obj_label": obj_label,
            "obj_pose": obj_pose,
            "verb": verb,
        }
