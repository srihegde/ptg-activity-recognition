'''
Adapted from: https://github.com/longcw/yolo2-pytorch
(Including the folders 'layers/' and 'utils/' in this 'components/' directory)
'''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import network as net_utils
from .layers.reorg.reorg_layer import ReorgLayer


def _make_layers(in_channels, net_cfg, batchnorm):
    layers = []

    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        for sub_cfg in net_cfg:
            layer, in_channels = _make_layers(in_channels, sub_cfg)
            layers.append(layer)
    else:
        for item in net_cfg:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, ksize = item
                if batchnorm:
                    layers.append(net_utils.Conv2d_BatchNorm(in_channels,
                                                             out_channels,
                                                             ksize,
                                                             same_padding=True))
                else:
                    layers.append(net_utils.Conv2d(in_channels, 
                                                    out_channels,
                                                    ksize,
                                                    same_padding=True))
                in_channels = out_channels

    return nn.Sequential(*layers), in_channels


class Darknet19(nn.Module):
    def __init__(self, num_cpts: int, obj_classes: int, verb_classes: int, batchnorm: bool = True):
        super(Darknet19, self).__init__()

        net_cfgs = [
            # conv1_set
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
            # conv2
            ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
            # ------------
            # conv3
            [(1024, 3), (1024, 3)],
            # conv4
            [(64, 1)],
            # conv 5
            [(1024, 3)]
        ]

        # darknet
        self.conv1_set, c1 = _make_layers(3, net_cfgs[0:5])
        self.conv2, c2 = _make_layers(c1, net_cfgs[5])
        # ---
        self.conv3, c3 = _make_layers(c2, net_cfgs[6])
        self.conv4, c4 = _make_layers(c1, net_cfgs[7])

        stride = 2
        # stride*stride times the channels of conv4
        self.reorg = ReorgLayer(stride=2)
        # cat [conv4, conv3]
        self.conv5, c5 = _make_layers((c4*(stride*stride) + c3), net_cfgs[8])

        # elements in 'out_channels' dimension depicts the flattened grid
        out_channels = 5*2*(3*num_cpts + 1 + obj_classes + verb_classes)
        # grid formation layer
        self.conv6 = net_utils.Conv2d(c5, out_channels, 1, 1, relu=False)

    def forward(self, x):
        conv1_set = self.conv1_set(x)
        conv2 = self.conv2(conv1_set)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv1_set)
        conv4_reorg = self.reorg(conv4)
        cat_4_3 = torch.cat([conv4_reorg, conv3], 1)
        conv5 = self.conv5(cat_4_3)
        grid = self.conv6(conv5)   # batch_size, out_channels, h, w

        return grid

    def load_from_npz(self, fname, num_conv=None):
        dest_src = {'conv.weight': 'kernel', 'conv.bias': 'biases',
                    'bn.weight': 'gamma', 'bn.bias': 'biases',
                    'bn.running_mean': 'moving_mean',
                    'bn.running_var': 'moving_variance'}
        params = np.load(fname)
        own_dict = self.state_dict()
        keys = list(own_dict.keys())

        for i, start in enumerate(range(0, len(keys), 5)):
            if num_conv is not None and i >= num_conv:
                break
            end = min(start+5, len(keys))
            for key in keys[start:end]:
                list_key = key.split('.')
                ptype = dest_src['{}.{}'.format(list_key[-2], list_key[-1])]
                src_key = '{}-convolutional/{}:0'.format(i, ptype)
                print((src_key, own_dict[key].size(), params[src_key].shape))
                param = torch.from_numpy(params[src_key])
                if ptype == 'kernel':
                    param = param.permute(3, 2, 0, 1)
                own_dict[key].copy_(param)


if __name__ == '__main__':
    net = Darknet19()
    # net.load_from_npz('models/yolo-voc.weights.npz')
    net.load_from_npz('models/darknet19.weights.npz', num_conv=18)
