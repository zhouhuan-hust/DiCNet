#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random
import collections
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch._six import string_classes, int_classes
from datasets.tools.tensor_helper import TensorHelper


def collate(batch, trans_dict， key="img_data"):
    """
        align image sizes by padding
    """
    data_keys = batch[0].keys()

    target_width, target_height = trans_dict['input_size']

    for i in range(len(batch)):
        channels, height, width = batch[i][key].size()
        if height == target_height and width == target_width:
            continue

        scaled_size = [width, height]

        w_scale_ratio = target_width / width
        h_scale_ratio = target_height / height
        w_scale_ratio = min(w_scale_ratio, h_scale_ratio)
        h_scale_ratio = w_scale_ratio
       
        scaled_size = (int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio)))
        scaled_size_hw = (scaled_size[1], scaled_size[0])
        batch[i][key] = TensorHelper.resize(batch[i][key].data,
                                                            scaled_size_hw, mode='bilinear', align_corners=True)
        if 'labelmap' in data_keys:
            batch[i]['labelmap'] = TensorHelper.resize(batch[i]['labelmap'].data,
                                                                     scaled_size_hw, mode='nearest')
        if 'maskmap' in data_keys:
            batch[i]['maskmap'] = TensorHelper.resize(batch[i]['maskmap'].data,
                                                                            scaled_size_hw, mode='nearest')
                                            
        pad_width = target_width - scaled_size[0]
        pad_height = target_height - scaled_size[1]
        assert pad_height >= 0 and pad_width >= 0
        # if pad_width > 0 or pad_height > 0:
        left_pad, up_pad = None, None
        if 'pad_mode' not in trans_dict or trans_dict['pad_mode'] == 'random':
            left_pad = random.randint(0, pad_width)  # pad_left
            up_pad = random.randint(0, pad_height)  # pad_up
        elif trans_dict['pad_mode'] == 'pad_border':
            direction = random.randint(0, 1)
            left_pad = pad_width if direction == 0 else 0
            up_pad = pad_height if direction == 0  else 0
        elif trans_dict['pad_mode'] == 'pad_left_up':
            left_pad = pad_width
            up_pad = pad_height
        elif trans_dict['pad_mode'] == 'pad_right_down':
            left_pad = 0
            up_pad = 0
        elif trans_dict['pad_mode'] == 'pad_center':
            left_pad = pad_width // 2
            up_pad = pad_height // 2
        else:
            Log.error('Invalid pad mode: {}'.format(trans_dict['pad_mode']))
            exit(1)
        pad = [left_pad, pad_width-left_pad, up_pad, pad_height-up_pad]
        batch[i][key] = F.pad(batch[i][key].data, pad=pad, value=0)
       
        if 'labelmap' in data_keys:
            batch[i]['labelmap'] = F.pad(batch[i]['labelmap'].data, pad=pad, value=-1)
        if 'maskmap' in data_keys:
            batch[i]['maskmap'] = F.pad(batch[i]['maskmap'].data, pad=pad, value=-1)

     return default_collate(batch)


if __name__ == "__main__":
    batch = [{"x1":torch.rand(3, 224, 224), "x2":torch.rand(3, 224, 224), 'maskmap':torch.rand(224, 224), 'labelmap':torch.rand(224, 224)},
            {"x1":torch.rand(3, 200, 220), "x2":torch.rand(3, 200, 220), 'maskmap':torch.rand(200, 220), 'labelmap':torch.rand(200, 220)}
    ]
    trans_dict = {"input_size": (256, 256), "pad_mode": "pad_center"}

    x = collate(batch, trans_dict)
    print(x["x1"].size())