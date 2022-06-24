# System libs
import os
import time
# import math
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
from models.teacher import T, T2
import torch.nn as nn
from collections import OrderedDict
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback

from easydict import EasyDict
from dataset import TrainDataset, ValDataset

opt = EasyDict()
opt.segm_downsampling_rate = 8
opt.imgSizes = (300, 375, 450, 525, 600)
opt.imgMaxSize = 1000
opt.padding_constant = 8
opt.segm_downsampling_rate = 8
opt.num_gpus = len(cfg.use_gpus)

def create_teacher(cfg):
    encoder = ModelBuilder.build_encoder(cfg.t_arch, 2048, cfg.teacher_weight[0])
    decoder = ModelBuilder.build_decoder("ppm_deepsup", fc_dim=2048, num_class=13, weights=cfg.teacher_weight[1])
    seg_model = nn.Sequential(OrderedDict([("net_enc", encoder), ("net_dec", decoder)]))
    bn_param = (decoder.conv_last[1].running_mean.cuda(), decoder.conv_last[1].running_var.cuda())
    model = T(seg_model)
    if cfg.use_gpus.__len__() > 1:
        model = UserScatteredDataParallel(
                    model,
                    device_ids=list(range(len(cfg.use_gpus))))
        patch_replication_callback(model)
    model = model.cuda()
    return model, bn_param

def create_student(cfg, bn_param):
    encoder = ModelBuilder.build_encoder(cfg.s_arch, 512, cfg.student_weight[0], pretrained=False)
    decoder = ModelBuilder.build_decoder("ppm_deepsup", fc_dim=512, num_class=13)
    seg_model = nn.Sequential(OrderedDict([("net_enc", encoder), ("net_dec", decoder)]))
    model = T2(seg_model)

    if len(cfg.weight_student) > 0:
        param_dict = torch.load(cfg.weight_student, map_location="cpu")
        keys = list(param_dict.keys())
        param_dict = OrderedDict({k.replace("module.", ""): param_dict.pop(k) for k in keys })
        model.load_state_dict(param_dict)
        print(">>> resume from  %s" % cfg.weight_student)
    
    if cfg.use_gpus.__len__() > 1:
        model = UserScatteredDataParallel(
                    model,
                    device_ids=list(range(len(cfg.use_gpus))))
        patch_replication_callback(model)
    model = model.cuda()
    return model

def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
    else:
        print("Non-deterministic")

# train one epoch
def train_one_epoch(student, teacher, data_loader, optimizer, lr_scheduler, history, epoch, cfg, plot_logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()

    local_progress = tqdm(data_loader, desc=f"Epoch {epoch}/{cfg.TRAIN.max_epochs}", disable=cfg.TRAIN.hide_bar, dynamic_ncols=True)
    
    student.train()
    teacher.eval()
    # main loop
    tic = time.time()

    for i, batch_data in enumerate(local_progress):
        # load a batch of data
        data_time.update(time.time() - tic)
        student.zero_grad()

        if cfg.use_gpus.__len__() == 1:
            batch_data = batch_data[0]
            batch_data['img_data'] = batch_data['img_data'].cuda()

        with torch.no_grad():
            Tf = teacher(batch_data)
        Sf = student(batch_data)

        loss = nn.functional.mse_loss(Sf, Tf)
        loss.backward()
        optimizer.step()
        lr = lr_scheduler.step()
        cfg.TRAIN.running_lr = float(lr)
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        plot_logger.update({'epoch':epoch, 'lr':lr, 'loss':ave_total_loss.val})

        local_progress.set_postfix({"lr": cfg.TRAIN.running_lr, 
                                                    "loss": ave_total_loss.average()})

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.iter_per_epoch
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            plot_logger.save(os.path.join(cfg.DIR,'logger.svg'))


def checkpoint(model, history, cfg, epoch):
    # print('Saving checkpoints...')

    dict_encoder = model.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_encoder,
        '{}/student_epoch_{}.pth'.format(cfg.DIR, epoch))

def main(cfg):
    
    # Network Builders
    teacher, bn_param = create_teacher(cfg)
    student = create_student(cfg, bn_param)
    
    print(">>>>> config <<<<<<")
    print(cfg)

    assert (cfg.TRAIN.lr_decay) or (cfg.TRAIN.final_lr==cfg.TRAIN.base_lr)
    data_set = TrainDataset("/home/DiCNet/data", "data/train.odgt", opt, batch_per_gpu=cfg.batch_per_gpu)

    loader_train = torch.utils.data.DataLoader(
            data_set,
            batch_size=len(cfg.use_gpus),  # we have modified data_parallel
            shuffle=False,  # we do not use this param
            collate_fn=user_scattered_collate,
            num_workers=1,
            drop_last=True,
            pin_memory=True)

    # define optimizer
    optimizer = get_optimizer(
        cfg.TRAIN.optimizer, student, 
        lr=cfg.TRAIN.base_lr,
        momentum=cfg.TRAIN.momentum,   
        weight_decay=0.00001)


    lr_scheduler = LR_Scheduler(
        optimizer,
        cfg.TRAIN.warmup_epochs, cfg.TRAIN.warmup_lr,
        cfg.TRAIN.max_epochs, cfg.TRAIN.base_lr, cfg.TRAIN.final_lr, 
        len(loader_train)
    )

    history = {'train': {'epoch': [], 'loss': []}}
    plot_logger = PlotLogger(params=['epoch', 'lr', 'loss'])
    cfg.TRAIN.iter_per_epoch = len(loader_train)
    global_progress = tqdm(range(cfg.TRAIN.start_epoch, cfg.TRAIN.max_epochs), desc="Training")
    for epoch in global_progress:
    # for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.max_epochs):
        train_one_epoch(student, teacher, loader_train, optimizer, lr_scheduler, history, epoch+1, cfg, plot_logger)
        checkpoint(student, history, cfg, epoch+1)

    print('Training Done!')

if __name__ == '__main__':
    cfg.use_gpus = [0, 1, 2, 3]
    cfg.batch_per_gpu = 4
    cfg.TRAIN.start_epoch = 0   
    cfg.TRAIN.base_lr = 0.0001
    cfg.TRAIN.final_lr = 0.0001
    cfg.DIR = "ckpt/adamshpre34"
    cfg.TRAIN.optimizer = "adam"
    cfg.s_arch = "resnet34dilated"
    cfg.TRAIN.max_epochs = 200

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.weight_student = os.path.join(
            cfg.DIR, 'student_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.weight_student), "checkpoint does not exitst!"
    else:
        cfg.weight_student = ""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.use_gpus).replace("[", "").replace("]", "").replace(" ", "")

    cfg.TRAIN.running_lr = cfg.TRAIN.base_lr

    set_deterministic(cfg.TRAIN.seed)

    main(cfg)

