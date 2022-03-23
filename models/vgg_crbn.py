import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear,BinarizeConv2d

class ConstRatioNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, ratio=1.1, is_relative_detach=True):
        super().__init__()
        self.ratio = ratio
        self.is_relative_detach = is_relative_detach
        # self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.ratio != 0 and self.ratio != 1:
            scale = self.ratio * x.detach() if self.is_relative_detach else self.ratio * x
            x = scale
        return x


class VGG_CRBN(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG_CRBN, self).__init__()
        self.infl_ratio=1;
        self.ratio = 1.15
        self.features = nn.Sequential(
            BinarizeConv2d(3, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1,
                      bias=True),
            ConstRatioNoise(self.ratio),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            ConstRatioNoise(self.ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            ConstRatioNoise(self.ratio),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            ConstRatioNoise(self.ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            ConstRatioNoise(self.ratio),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=True),
            ConstRatioNoise(self.ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True)

        )
        self.classifier = nn.Sequential(
            BinarizeLinear(512 * 4 * 4, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(1024, num_classes, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


def vgg_crbn(**kwargs):
    num_classes = kwargs.get( 'num_classes', 10)
    return VGG_CRBN(num_classes)
