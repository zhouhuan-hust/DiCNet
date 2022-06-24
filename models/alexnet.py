import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels=1024, pool_factors=(1, 2, 3, 6), batch_norm=True):
        super().__init__()
        self.spatial_blocks = []
        for pf in pool_factors:
            self.spatial_blocks += [self._make_spatial_block(in_channels, pf, batch_norm)]
        self.spatial_blocks = nn.ModuleList(self.spatial_blocks)

        bottleneck = []
        bottleneck += [nn.Conv2d(in_channels * (len(pool_factors) + 1), out_channels, kernel_size=1)]
        # if batch_norm:
        #     bottleneck += [nn.BatchNorm2d(out_channels)]
        # bottleneck += [nn.ReLU(inplace=True)]
        self.bottleneck = nn.Sequential(*bottleneck)

    def _make_spatial_block(self, in_channels, pool_factor, batch_norm):
        spatial_block = []
        spatial_block += [nn.AdaptiveAvgPool2d(output_size=(pool_factor, pool_factor))]
        spatial_block += [nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)]
        if batch_norm:
            spatial_block += [nn.BatchNorm2d(in_channels)]
        spatial_block += [nn.ReLU(inplace=True)]

        return nn.Sequential(*spatial_block)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pool_outs = [x]
        for block in self.spatial_blocks:
            pooled = block(x)
            pool_outs += [F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)]
        o = torch.cat(pool_outs, dim=1)
        o = self.bottleneck(o)
        return o

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class PSPnet(torch.nn.Module):

    def __init__(self, batch_norm=True, in_channels=256, psp_out_feature=1024):
        super(PSPnet, self).__init__()
        self.PSP = PSPModule(in_channels=in_channels, out_channels=psp_out_feature, batch_norm=batch_norm)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):            
        o = self.PSP(x)

        return o




class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            #nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )
        self.PSPnet=PSPnet(batch_norm=True, in_channels=256, psp_out_feature=512)

    def forward(self, x):
        x = self.features(x)
        out = self.PSPnet(x)

        #x = x.view(x.size(0), 256 * 6 * 6)
        #x = self.classifier(x)
        return out





# model = AlexNet()
# model.load_state_dict(model_zoo.load_url(model_urls['alexnet']), strict=False)
# input = torch.randn(2,3,320,320)
# output = model(input)
# print('####1111', input.shape, output.shape)