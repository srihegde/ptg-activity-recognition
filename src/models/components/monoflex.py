"""

TODO:
* Integrate hydra configs and remove monoflex configs

"""

from torch import nn

from .backbone import build_backbone
from .config import cfg
from .head.detector_head import bulid_head
from .structures.image_list import to_image_list


class KeypointDetector(nn.Module):
    """Generalized structure for keypoint based object detector. main parts:

    - backbone
    - heads
    """

    def __init__(self, test):
        super(KeypointDetector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.heads = bulid_head(cfg, self.backbone.out_channels)
        self.test = test

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        features = self.backbone(images.tensors)

        if self.training:
            loss_dict, log_loss_dict = self.heads(features, targets)
            return loss_dict, log_loss_dict
        else:
            result, eval_utils, visualize_preds = self.heads(features, targets, test=self.test)
            return result, eval_utils, visualize_preds
