from easydict import EasyDict

cfg = EasyDict()
cfg.DIR = "ckpt/bddresnet34" 
cfg.t_arch = "resnet50dilated"
cfg.teacher_weight = ("./pretrained/encoder.pth", "./pretrained/decoder.pth")
cfg.student_weight = ("./pretrained/resnet34-imagenet.pth", "./pretrained/resnet18-imagenet.pth", "./pretrained/resnet50-imagenet.pth")
cfg.s_arch = "resnet34dilated"
cfg.batch_per_gpu = 4
cfg.weight_student = ""
cfg.use_gpus = [0,1,2,3]

cfg.TRAIN = EasyDict()
cfg.TRAIN.optimizer = "adam"
cfg.TRAIN.base_lr = 0.01
cfg.TRAIN.momentum = 0.9
cfg.TRAIN.weight_decay = 1e-4
cfg.TRAIN.warmup_lr = 0.00001
cfg.TRAIN.warmup_epochs = 0
cfg.TRAIN.final_lr = 0.01
cfg.TRAIN.start_epoch = 80
cfg.TRAIN.max_epochs = 200
cfg.TRAIN.seed = 666
cfg.TRAIN.hide_bar = False
cfg.TRAIN.disp_iter = 10
cfg.TRAIN.lr_decay = False
