# System libs
import os
import time
import math
import random
import argparse
# Numerical libs
import torch
import numpy as np
import torch.nn as nn
# Our libs
from config_student import cfg
from utils import AverageMeter, setup_logger
from optimizers import get_optimizer, LR_Scheduler
from plot_logger import PlotLogger
from tqdm import tqdm

from models.model import ModelBuilder
from models.teacher import T,T2
import torch.nn as nn
from collections import OrderedDict

from easydict import EasyDict
from dataset import TrainDataset, ValDataset

from anom_utils import eval_ood_measure
import matplotlib.pyplot as plt

opt = EasyDict()
opt.segm_downsampling_rate = 8
opt.imgSizes = (300, 375, 450, 525, 600)
opt.imgMaxSize = 1000
opt.padding_constant = 8
opt.segm_downsampling_rate = 8
opt.num_gpus = len(cfg.use_gpus)

from scipy.io import loadmat
from utils import colorEncode
colors = loadmat('data/color150.mat')['color_new']


def create_teacher(cfg):
    encoder = ModelBuilder.build_encoder(cfg.t_arch, 2048, cfg.teacher_weight[0])
    decoder = ModelBuilder.build_decoder("ppm_deepsup", fc_dim=2048, num_class=13, weights=cfg.teacher_weight[1])
    seg_model = nn.Sequential(OrderedDict([("net_enc", encoder), ("net_dec", decoder)]))
    bn_param = (decoder.conv_last[1].running_mean.cuda(), decoder.conv_last[1].running_var.cuda())
    model = T(seg_model)
    
    class Conv(nn.Module):
        def __init__(self, weight, bias):
            super().__init__()
            self.weight = weight
            self.bias = bias
            
        def forward(self, x):
            return x * self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) +self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
    class Module(nn.Module):
        def __init__(self, model, classifier):
            super().__init__()
            self.model = model
            self.classifier = classifier
        
        def forward(self, x):
            x1 = self.model(x)
            x2 = self.classifier(x1)
            return x1, x2

    classifier = nn.Sequential(Conv(decoder.conv_last[1].weight, decoder.conv_last[1].bias),
                            nn.ReLU(inplace=True),
                            decoder.conv_last[4])

    teacher = Module(model, classifier)
                                    
    return teacher.cuda(), bn_param

def create_student(cfg, bn_param):
    encoder = ModelBuilder.build_encoder(cfg.s_arch, 512,  pretrained=False)
    decoder = ModelBuilder.build_decoder("ppm_deepsup", fc_dim=512, num_class=13)
    seg_model = nn.Sequential(OrderedDict([("net_enc", encoder), ("net_dec", decoder)]))
    model = T2(seg_model)
    
    if len(cfg.weight_student) > 0:
        param_dict = torch.load(cfg.weight_student, map_location="cpu")
        keys = list(param_dict.keys())
        param_dict = OrderedDict({k.replace("module.", ""): param_dict.pop(k) for k in keys })
        model.load_state_dict(param_dict)
        print(">>> load from epoch %d" % cfg.test_epoch)
    return model.cuda()

def t2np(t):
    return t.detach().cpu().numpy()

def conmpute_metric(score_map, label_map, mask=None):
    """
        score_map: HxW, numpy.array
        label_map: HxW, numpy.array
        mask:      HxW, numpy.array bool
    """
    if mask is not None:
        score_map = score_map[mask]
        label_map = label_map[mask]
    auroc, aupr, fpr, thr = eval_ood_measure(score_map,label_map)
    return {"auroc": auroc, "aupr": aupr, "fpr95": fpr}, thr

def cal_entropy(logits_batch):
    N, H, W = logits_batch.size(0), logits_batch.size(2), logits_batch.size(3)
    softmax_batch = nn.functional.softmax(logits_batch/5, dim=1)
    softmax_batch = softmax_batch.cpu().numpy()
    entropy_logits = np.zeros((N,H,W))
    for n in range(N):
        for h in range(H):
            for w in range(W):
                entropy_logits[n,h,w] = sum([-p*math.log(p, 2) for p in softmax_batch[n,:,h,w]])

    return torch.from_numpy(entropy_logits)

class Logger(object):
    def __init__(self, save_dir="eval_dir"):
        self.save_dir = save_dir
        self.logger = {"filename":[], "auroc":[], "aupr":[], "fpr95": [], "efpr":[], "cfpr":[]}

    def add(self, metric_dict):
        for k, v in metric_dict.items():
            self.logger[k].append(v)
    
    def print_metric(self):
        print(">>>> evaluation results <<<<<")
        for k in ["auroc", "aupr", "fpr95", "efpr", "cfpr"]:
            print(f"{k}: {self.get(k):.4f}")

    def dump(self):
        import pandas as pd
        df = pd.DataFrame.from_dict(self.logger)
        df.to_excel(f"{self.save_dir}/logger.xlsx")

    def get(self, k):
        return np.array(self.logger[k]).mean()

def visualization(img_ori, label_map, Tf, Sf, score_map, info, score_thresh, res, logits):
    from utils import plot_images
    file_path = "./eval_dir/visualization"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    filename = info.split("/")[-1]
    pred = np.zeros_like(score_map, np.uint8)
    text = "aupr:%.3f, fpr:%.3f, auroc:%.3f" % (res["aupr"], res["fpr95"], res["auroc"])
    pred[score_map>score_thresh] = 255
    ano_label = np.zeros_like(label_map, dtype=np.uint8)
    ano_label[label_map==13] = 255

    logits = logits.argmax(0)
    pred_color = colorEncode(logits, colors)
    label_color = colorEncode(label_map, colors)

    plot_images([img_ori, ano_label, Tf, Sf, score_map, pred, label_color, pred_color], (320, 180), (4, 2), os.path.join(file_path, filename), text)

def main(cfg):
    # Network Builders
    teacher, bn_param = create_teacher(cfg)
    student = create_student(cfg, bn_param)

    if torch.cuda.device_count() > 1: 
        teacher = torch.nn.DataParallel(teacher)
        student = torch.nn.DataParallel(student)

    data_test = ValDataset("/home/DiCNet/data", "data/test.odgt", opt)

    data_loader = torch.utils.data.DataLoader(
            data_test,
            batch_size=len(cfg.use_gpus) * cfg.batch_per_gpu, 
            shuffle=False, 
            num_workers=8,
            drop_last=False,
            pin_memory=True)

    student.eval()
    teacher.eval()
    # main loop
    local_progress = tqdm(data_loader, dynamic_ncols=True)
    logger = Logger()

    for i, batch_data in enumerate(local_progress):
        # load a batch of data
        batch_data['img_data'] = batch_data['img_data'].cuda()

        with torch.no_grad():
            Tf, logits_batch = teacher(batch_data)
            Sf = student(batch_data)

        score_map = torch.pow(Tf - Sf, 2).mean(dim=1)

        batch_data["seg_label"].squeeze_(1)
        score_map = nn.functional.interpolate(score_map.unsqueeze(1), batch_data["seg_label"].shape[-2:], mode="bilinear", align_corners=True)
        score_map = (score_map - torch.min(score_map))/ (torch.max(score_map) - torch.min(score_map))
        score_map.squeeze_(1)

        logits_batch = nn.functional.interpolate(logits_batch, (720, 1280), mode="bilinear", align_corners=False)
        for s, l, tf, sf, o, logits, info in zip(t2np(score_map), t2np(batch_data["seg_label"]), t2np(Tf).mean(axis=1),
                                                     t2np(Sf).mean(axis=1), t2np(batch_data["img_ori"]), t2np(logits_batch), batch_data["info"]):
            res, thr = conmpute_metric(s, l)

            res["filename"] = info
            logger.add(res)

            pred = logits.argmax(axis=0)
            seg_label = l
            ano_mask = l < 13 
            cmask = np.logical_and(pred == seg_label, ano_mask)
            wmask = np.logical_and(~cmask, ano_mask)

            binary_map = s > thr
            cfpr = binary_map[cmask].sum() / cmask.sum()
            efpr = binary_map[wmask].sum() / wmask.sum()
            #cfpr = binary_map[cmask].sum() / float(cmask.sum()+1e-6)
            #efpr = binary_map[wmask].sum() / float(wmask.sum()+1e-6)

            logger.add({"efpr":efpr, "cfpr":cfpr})
            # print(">>> ", cfpr, efpr)

            if cfg.vis:
                visualization(o, l, tf, sf, s, info, 0.8, res, logits)

        local_progress.set_postfix({"auroc": logger.get("auroc"), 
                                                    "aupr": logger.get("aupr"), "fpr95":logger.get("fpr95"), 
                                                    "efpr":logger.get("efpr"), "cfpr":logger.get("cfpr")})
    logger.print_metric() # print average metric
    logger.dump() # save an excel file as "eval_dir/logger.xlsx"

if __name__ == '__main__':

    from easydict import EasyDict

    cfg = EasyDict()
    cfg.DIR = "ckpt/adamshpre34"
    cfg.t_arch = "resnet50dilated"
    cfg.teacher_weight = ("./pretrained/encoder.pth", "./pretrained/decoder.pth")
    cfg.s_arch = "resnet34dilated"
    cfg.batch_per_gpu = 1
    cfg.weight_student = ""
    cfg.use_gpus = [3]
    cfg.test_epoch = 115
    cfg.vis = False
    cfg.weight_student = os.path.join(
            cfg.DIR, 'student_epoch_{}.pth'.format(cfg.test_epoch))
    assert os.path.exists(cfg.weight_student), "checkpoint does not exitst!"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.use_gpus).replace("[", "").replace("]", "").replace(" ", "")

    main(cfg)

