"""

TODO:
* Separate transforms for training and testing
* Transform input frame to 416x416
* Use label_split info for VideoDataset
* Update documentation

"""

import pdb
from typing import Dict, Optional

# import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from .components.frame_dataset import H2OFrameDataset
from .components.video_dataset import H2OVideoDataset


class H2ODataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        pose_files: Dict,
        action_files: Dict,
        data_dir: str = "data/h2o",
        data_type: str = "video",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        frames_per_segment: int = 1
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.data_type = data_type
        self.frames_per_segment = frames_per_segment

        # data transformations
        if self.data_type == "frame":
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif self.data_type == "video":
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 37  # Action (interaction) classes

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`,
        `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and
        `trainer.test()`, so be careful not to execute the random split twice!
        The `stage` can be used to differentiate whether it's called before
        trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            if self.data_type == "frame":
                # pdb.set_trace()
                self.data_train = H2OFrameDataset(
                    self.hparams.data_dir,
                    self.hparams.pose_files["train_list"],
                    transform=self.transforms,
                )
                self.data_val = H2OFrameDataset(
                    self.hparams.data_dir,
                    self.hparams.pose_files["val_list"],
                    transform=self.transforms,
                )
                self.data_test = H2OFrameDataset(
                    self.hparams.data_dir,
                    self.hparams.pose_files["test_list"],
                    transform=self.transforms,
                )
            elif self.data_type == "video":
                self.data_train = H2OVideoDataset(
                    self.hparams.data_dir,
                    self.hparams.action_files["train_list"],
                    frames_per_segment = self.frames_per_segment,
                    transform=self.transforms,
                )
                self.data_val = H2OVideoDataset(
                    self.hparams.data_dir,
                    self.hparams.action_files["val_list"],
                    frames_per_segment = self.frames_per_segment,
                    transform=self.transforms,
                    test_mode=True,
                )
                self.data_test = H2OVideoDataset(
                    self.hparams.data_dir,
                    self.hparams.action_files["test_list"],
                    frames_per_segment = self.frames_per_segment,
                    transform=self.transforms,
                    test_mode=True,
                )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
