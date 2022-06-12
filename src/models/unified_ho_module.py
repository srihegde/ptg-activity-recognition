"""

TODO:
* Implement Unified FCN
* Implement Temporal Module
* Update documentation

"""

from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class UnifiedFCNModule(nn.Module):
    """Class implements fully convolutional network for extracting spatial features from the video
    frames.

    Args: TBD
    """

    def __init__(
        self, net: str, num_cpts: int, obj_classes: int, verb_classes: int, batchnorm: bool = True
    ):
        super(UnifiedFCNModule, self).__init__()
        self.net = self._select_network(net)

    def _select_network(self, net_opt: str) -> nn.Module:
        net: nn.Module = None
        if net_opt == "convnext_tiny":
            net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            print("NN model not found. Change the feature extractor network.")

        return net

    def forward(self, x: torch.Tensor):
        # g = self.net(x)
        # h_grid = g[...,]
        pass


class TemporalModule(nn.Module):
    """docstring for TemporalModule."""

    def __init__(self, arg):
        super(TemporalModule, self).__init__()
        self.arg = arg


class UnifiedHOModule(LightningModule):
    """This class implements the spatio-temporal model used for unified representation of hands and
    interacting objects in the scene. This model also performs the activity recognition for the
    given frame sequence.

    Args: TBD
    """

    def __init__(
        self,
        fcn: torch.nn.Module,
        temporal: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.fcn = fcn
        self.temporal = temporal

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.fcn(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log(
            "val/acc_best",
            self.val_acc_best.compute(),
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
