import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import BinarizeLinear, BinarizeConv2d, BinarizeAdaptiveAvgPool2d


class VGG16_GAP(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG16_GAP, self).__init__()
        self.infl_ratio = 1
        self.features = nn.Sequential(
            BinarizeConv2d(3, 64*self.infl_ratio, kernel_size=3, stride=1, padding=1,
                           bias=True),
            nn.BatchNorm2d(64*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(64*self.infl_ratio, 64*self.infl_ratio,
                           kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(64, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1,
                           bias=True),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio,
                           kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio,
                           kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio,
                           kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio,
                           kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(256*self.infl_ratio, 512*self.infl_ratio,
                           kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(512*self.infl_ratio, 512*self.infl_ratio,
                           kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(512*self.infl_ratio, 512,
                           kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(512*self.infl_ratio, 512*self.infl_ratio,
                           kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(512*self.infl_ratio, 512*self.infl_ratio,
                           kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(512*self.infl_ratio, 512,
                           kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),
            
            BinarizeConv2d(512*self.infl_ratio, num_classes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.Hardtanh(inplace=True),

        )
        self.classifier = nn.Sequential(
            # BinarizeLinear(512 * 4 * 4, 1024, bias=True),
            # nn.BatchNorm1d(1024),
            # nn.Hardtanh(inplace=True),
            # # nn.Dropout(0.5),
            # BinarizeLinear(1024, 1024, bias=True),
            # nn.BatchNorm1d(1024),
            # nn.Hardtanh(inplace=True),
            # # nn.Dropout(0.5),
            # BinarizeLinear(1024, num_classes, bias=True),
            # nn.BatchNorm1d(num_classes, affine=False),
            # nn.LogSoftmax()
            
            BinarizeAdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(num_classes, affine=False),
            nn.LogSoftmax()
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5},
            160: {'lr': 5e-6},
            180: {'lr': 1e-6},
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


def vgg16_gap(**kwargs):
    num_classes = kwargs.get('num_classes', 10)
    return VGG16_GAP(num_classes)
