
import torch
import torch.nn as nn
from collections import OrderedDict
# self.decoder.conv_last[1].running_mean
# self.decoder.conv_last[1].running_var
class T(nn.Module):
    def __init__(self, seg_model, bn_param=None):
        super().__init__()
        self.encoder = seg_model.net_enc
        self.decoder = seg_model.net_dec
        self.decoder = nn.Sequential(OrderedDict([("ppm", seg_model.net_dec.ppm),
                                                ("conv_last", seg_model.net_dec.conv_last)]))
        self.decoder.conv_last[1].affine = False
        self.decoder.conv_last = nn.Sequential(self.decoder.conv_last[0], # conv
                                                self.decoder.conv_last[1]) # bn
        self.running_mean, self.running_var = bn_param if bn_param else (None, None)

    def forward(self, feed_dict):
        x = self.encoder(feed_dict['img_data'], return_feature_maps=True)
        x = self.forward_decoder(x)
        return x

    def forward_decoder(self, conv_out, segSize=None):

        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.decoder.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        if self.running_mean is not None:
            x = self.decoder.conv_last[0](ppm_out)
            x = (x - self.running_mean.unsqueeze(-1).unsqueeze(-1).to(x.device)) / (self.running_var.unsqueeze(-1).unsqueeze(-1).to(x.device) + 1e-5)
        else:
            x = self.decoder.conv_last(ppm_out)

        return x

class T2(nn.Module):
    def __init__(self, seg_model, bn_param=None):
        super().__init__()
        self.encoder = seg_model.net_enc
        self.decoder = seg_model.net_dec
        self.decoder = nn.Sequential(OrderedDict([("ppm", seg_model.net_dec.ppm),
                                                ("conv_last", seg_model.net_dec.conv_last)]))
        self.decoder.conv_last[1].affine = False
        self.decoder.conv_last = nn.Sequential(self.decoder.conv_last[0]) # conv
        self.running_mean, self.running_var = bn_param if bn_param else (None, None)

    def forward(self, feed_dict):
        x = self.encoder(feed_dict['img_data'], return_feature_maps=True)
        x = self.forward_decoder(x)
        return x

    def forward_decoder(self, conv_out, segSize=None):

        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.decoder.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        if self.running_mean is not None:
            x = self.decoder.conv_last[0](ppm_out)
            x = (x - self.running_mean.unsqueeze(-1).unsqueeze(-1).to(x.device)) / (self.running_var.unsqueeze(-1).unsqueeze(-1).to(x.device) + 1e-5)
        else:
            x = self.decoder.conv_last(ppm_out)

        return x