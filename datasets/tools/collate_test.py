#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random
import collections
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch._six import string_classes, int_classes
from datasets.tools.tensor_helper import TensorHelper

def subcollate(batch, trans_dict):
    """
    docstring
    """
    assert len(batch) == 1
    batch = batch[0]
    data_keys = batch.keys()

    # 对于x1, 保持长宽比scale以匹配目标尺寸，再做padding
    target_width, target_height = trans_dict['input_size']
    channels, height, width = batch["x1"].size()
    if height == target_height and width == target_width: 
        pass
    else:
        scaled_size = [width, height]
        w_scale_ratio = target_width / width
        h_scale_ratio = target_height / height
        w_scale_ratio = min(w_scale_ratio, h_scale_ratio)
        h_scale_ratio = w_scale_ratio
        scaled_size = (int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio)))
        scaled_size_hw = (scaled_size[1], scaled_size[0])
        
        pad_width = target_width - scaled_size[0]
        pad_height = target_height - scaled_size[1]
        assert pad_height >= 0 and pad_width >= 0
        
        left_pad = pad_width // 2
        up_pad = pad_height // 2

        pad = [left_pad, pad_width-left_pad, up_pad, pad_height-up_pad]
        
        
        batch["x1"] = TensorHelper.resize(batch["x1"].data,
                                                            scaled_size_hw, mode='bilinear', align_corners=True)
        batch["x1"] = F.pad(batch["x1"].data, pad=pad, value=0)

        if 'labelmap' in data_keys:
            batch['labelmap'] = TensorHelper.resize(batch['labelmap'].data,
                                                                     scaled_size_hw, mode='nearest')
            batch['labelmap'] = F.pad(batch['labelmap'].data, pad=pad, value=-1)

        if 'maskmap' in data_keys:
            batch['maskmap'] = TensorHelper.resize(batch['maskmap'].data,
                                                                        scaled_size_hw, mode='nearest')
            batch['maskmap'] = F.pad(batch['maskmap'].data, pad=pad, value=1)                                                            
    batch['pos'] = []
    # 对于x2, 如果超出目标尺寸，则缩小再做padding
    for i, image2 in enumerate(batch["x2"]):
        channels, height, width = image2.size()
        scaled_size = [width, height]
        w_scale_ratio = target_width / width
        h_scale_ratio = target_height / height
        w_scale_ratio = min(w_scale_ratio, h_scale_ratio)
        h_scale_ratio = w_scale_ratio

        if h_scale_ratio < 1.0: # 当前尺寸大于目标尺寸
            scaled_size = (int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio)))
            scaled_size_hw = (scaled_size[1], scaled_size[0])
            image2 = TensorHelper.resize(image2.data,
                                        scaled_size_hw, mode='bilinear', align_corners=True)
                                        
        pad_width = target_width - scaled_size[0]
        pad_height = target_height - scaled_size[1]
        assert pad_height >= 0 and pad_width >= 0
        
        left_pad = pad_width // 2
        up_pad = pad_height // 2

        pad = [left_pad, pad_width-left_pad, up_pad, pad_height-up_pad]
        image2 = F.pad(image2.data, pad=pad, value=0)
        batch["x2"][i] = image2

        if "fea_stride" in trans_dict:
            top_pad = pad_width-left_pad
            bottom_pad = pad_height-up_pad
            stride = trans_dict["fea_stride"]
            batch["pos"].append((top_pad // stride, left_pad // stride, 
                            (scaled_size[1]+top_pad) // stride, (scaled_size[0]+left_pad) // stride))

    batch["x1"] = batch["x1"].unsqueeze(0)
    batch["x2"] = torch.cat([img.unsqueeze(0) for img in batch["x2"]], dim=0)
    batch['maskmap'] = batch['maskmap'].unsqueeze(0)
    batch['labelmap'] = batch['labelmap'].unsqueeze(0)
    batch['pos'] = torch.tensor(batch['pos'])

    return batch

def collate_test(batch, trans_dict):
    """
        align image sizes by padding
    """
    
    batch = subcollate(batch, trans_dict)

    return batch

if __name__ == "__main__":
    batch = [{"x1":torch.rand(3, 224, 224), "x2":torch.rand(3, 224, 224), 'maskmap':torch.rand(224, 224), 'labelmap':torch.rand(224, 224)},
            {"x1":torch.rand(3, 200, 220), "x2":torch.rand(3, 200, 220), 'maskmap':torch.rand(200, 220), 'labelmap':torch.rand(200, 220)}
    ]
    trans_dict = {"input_size": (256, 256), "pad_mode": "pad_center"}

    x = collate(batch, trans_dict)
    print(x["x1"].size())