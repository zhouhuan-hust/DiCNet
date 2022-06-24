import torch
from torch.utils import data
from datasets.default_loader import DefaultLoader
from datasets.test_loader import TestLoader
from datasets.tools.aug_transform import CV2AugCompose
import datasets.tools.transforms as transforms
from datasets.tools import collate, collate_test

img_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(255., mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])

class Dataloader(object):

    def __init__(self, opt=None):
        self.opt = opt
        self.data_root = "./data" if self.opt is None else opt.DATASET.data_root
                        
    def get_trainloader(self, ):
        if self.opt is not None:
            trainloader = data.DataLoader(DefaultLoader("train", self.data_root, aug_transform=CV2AugCompose(self.opt, "train"), 
                                                            img_transform=img_transform,
                                                            batch_size=self.opt.TRAIN.batch_size, shuffle=True, num_workers=4, 
                                                            collate_fn=lambda batch: collate(batch, self.opt.TRAIN), drop_last=True)
        else:
            trainloader = data.DataLoader(DefaultLoader("train", self.data_root), batch_size=4, 
                                                            shuffle=True, num_workers=4, 
                                                            collate_fn=lambda batch: collate(batch, {"input_size": (720, 720), "pad_mode": "pad_center", "fea_stride": 8}))                                               
        return trainloader

    def get_valloader(self):
        if self.opt is not None:
            valloader = data.DataLoader(DefaultLoader("val", self.data_root, aug_transform=CV2AugCompose(self.opt, "train"), 
                                                            img_transform=img_transform, jitter=CV2AugCompose(self.opt, "jitter")),
                                                            batch_size=self.opt.TRAIN.batch_size, shuffle=False, num_workers=4, 
                                                            collate_fn=lambda batch: collate(batch, self.opt.TRAIN))
        else:
            valloader = data.DataLoader(DefaultLoader("val", self.data_root), batch_size=4, 
                                                            shuffle=False, num_workers=4, 
                                                            collate_fn=lambda batch: collate(batch, {"input_size": (720, 720), "pad_mode": "pad_center", "fea_stride": 8}))
        return valloader

    def get_testloader(self):
        if self.opt is not None:
            testloader = data.DataLoader(TestLoader(self.data_root, aug_transform=None, img_transform=img_transform, num_x2=self.opt.TEST.num_x2,
                                                            jitter=CV2AugCompose(self.opt, "jitter")),
                                                            batch_size=1, shuffle=False, num_workers=4, 
                                                            collate_fn=lambda batch: collate_test(batch, self.opt.TEST))
        else:
            testloader = data.DataLoader(TestLoader(self.data_root, aug_transform=None, num_x2=8), 
                                                            batch_size=1, shuffle=False, num_workers=4, 
                                                            collate_fn=lambda batch: collate_test(batch, {"input_size": (1280, 720), "pad_mode": "pad_center", "fea_stride": 8}))
        return testloader

if __name__ == "__main__":
        loader = Dataloader()
        loader_train = loader.get_trainloader()
        for i, batch in enumerate(loader_train):
            print(batch.keys())
            if i > 10:
                print(batch)
                break