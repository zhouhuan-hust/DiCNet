import os
import os.path as osp
import numpy as np
from torch.utils import data
import cv2 
import json
import torch
from datasets.tools.aug_transform import CV2AugCompose
from datasets.tools.image_helper import ImageHelper
from datasets.tools.collate import collate
import datasets.tools.transforms as transforms
from datasets.tools import collate_test
from easydict import EasyDict
from copy import deepcopy

configer = EasyDict()
configer.DATASET = EasyDict()
configer.DATASET.input_mode = "RGB"
configer.DATASET.aug_transform = EasyDict()
_C = configer.DATASET.aug_transform
_C.train = {"aug_keys":["random_crop"],
        "random_crop":{
          "ratio":     1.0,
          "crop_size": [700, 700],
          "method":    "random",
          "allow_outside_center": False
        }
      }
_C.jitter = { "aug_keys": ["random_resize", "random_hflip", "random_brightness", 
                            "random_contrast", "random_saturation", "random_hue"],
              "random_resize":{
                "ratio":        1.0,
                "method":       "random",
                "scale_range":  [0.5, 1.2],
                "aspect_range": [0.9, 1.1] },
              "random_hflip":       {"swap_pair": None, "ratio": 0.5},
              "random_brightness":  {"shift_value": 30, "ratio": 0.5},
              "random_contrast":    {"lower": 0.5,      "upper": 1.5, "ratio": 0.8},
              "random_saturation":  {"lower": 0.5,      "upper": 1.5, "ratio": 0.8},
              "random_hue":         {"delta": 18,       "ratio": 0.8}}
            
DefaultLabelTransform = transforms.Compose([
            transforms.ToLabel(),
            transforms.ReLabel(255, -1)])

DefaultImgTransform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(255.0, mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
DefaultAugTransform = CV2AugCompose(configer, "train")
DefaultJitter = CV2AugCompose(configer, "jitter")

class TestLoader(data.Dataset):
    def __init__(self, data_root="./data", aug_transform=None, num_x2=1,
                    img_transform=DefaultImgTransform, label_transform=DefaultLabelTransform, 
                                    jitter=DefaultJitter, input_mode="RGB", opt=None, *kwargs):
        odgt = osp.join(data_root, "odgt_street", "test.odgt")
        self.list_sample = None
        self.data_root = data_root
        self.parse_input_list(odgt)
        self.jitter = jitter
        self.num_x2 = num_x2
        self.input_mode = input_mode
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
    

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        """
            Input: 
                odgt: file path, meta information dumped by Json
        """
        if isinstance(odgt, list):
            self.list_sample = odgt

        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')][0]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __len__(self):
        return len(self.list_sample)

    def __getitem__(self, index):

        meta_info = self.list_sample[index]
        image_path = os.path.join(self.data_root, meta_info['fpath_img'])
        segm_path = os.path.join(self.data_root, meta_info['fpath_segm'])
        image = ImageHelper.cv2_read_image(image_path, mode=self.input_mode) # RGB mode
        labelmap = ImageHelper.cv2_read_image(segm_path)

        if self.aug_transform is not None:
            image1, labelmap, _ = self.aug_transform(image, labelmap=labelmap)
        else:
            image1 = image

        image2_list = []
        flip_flag_list = []
        if self.jitter is not None:
            for i in range(self.num_x2):
                image2, flip_flag = self.jitter(image1.copy())
                image2_list.append(deepcopy(image2))
                flip_flag_list.append(deepcopy(flip_flag))

        if self.img_transform is not None:
            image1 = self.img_transform(image1)
            image2_list = [self.img_transform(img) for img in image2_list]
                
        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)[...,0]

        maskmap = torch.ones_like(labelmap, dtype=torch.uint8)

        return {'x1': image1, 'x2': image2_list, 'labelmap': labelmap, 'maskmap': maskmap, "flip_flag": torch.as_tensor(flip_flag_list)}

def print_dict_size(dic):
    print("---")
    for k, v in dic.items():
        print(k, "--> ", v.size())

if __name__ == "__main__":
    data_root = "./data"
    loader = data.DataLoader(TestLoader(data_root, num_x2=4),
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            collate_fn=lambda batch: collate_test(batch, {"input_size": (700, 700),
                                                                    "pad_mode": "pad_center",
                                                                    "fea_stride": 8}))
 
    for i, item in enumerate(loader):
        # print_dict_size(item)
        if i > 1:
            print(item)
            break
    