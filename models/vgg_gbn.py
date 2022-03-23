import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear,BinarizeConv2d

class GaussianNoise(nn.Module):
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
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).clone().float().normal_() * scale
            x = x + sampled_noise
        return x

class DifferentialNoise(nn.Module):

    def __init__(self, n=50, layers=64, is_relative_detach=True):
        super().__init__()

        self.n = n * int(layers / 64)
        self.layer = layers
        self.is_relative_detach = is_relative_detach

        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        # print(f"Layer {self.__class__.__name__}:  {x.size()}")

        xx, yy, w, h = x.size()

        assert(w == h)
        sampled_noise = self.noise.expand(*x.size()).clone().float()
        # sampled_noise = x.to('cpu')
        # print(w)

        sampled_noise = x


        for i in range(xx):
            for j in range(yy):
                long_t = sampled_noise[i, j, :, :].reshape([-1, 1])
                if self.layer == 64:
                    for t in range(0, w*h, 2):
                        if t + 1 < w*h:
                            long_t[t + 1] = long_t[t+1] - long_t[t] / self.n
                        # if i > xx - 2 and j > yy - 2:
                        #    print(f"layer: {self.layer}, and data: {long_t}")
                    # print(long_t.size(), w, h)
                if self.layer == 128:
                    for t in range(0, w*h, 3):
                        if t + 1 < w*h:
                            long_t[t + 1] = long_t[t+1] - long_t[t] / self.n
                        if t + 2 < w*h:
                            long_t[t + 2] = long_t[t+2] - (long_t[t] + long_t[t+1]) / self.n
                            # if i < 2 and j < 2:
                            #    print(f"layer: {self.layer}, and data: {long_t[t+1]} and {long_t[t+2]}")
                if self.layer == 256:
                    for t in range(0, w*h, 4):
                        if t + 1 < w*h:
                            long_t[t + 1] = long_t[t+1] - long_t[t] / self.n
                        if t + 2 < w*h:
                            long_t[t + 2] = long_t[t+2] - (long_t[t] + long_t[t+1]) / self.n
                        if i + 3 < w*h:
                            long_t[t + 3] = long_t[t+3] - (long_t[t] + long_t[t+1] + long_t[t+2]) / self.n
                if self.layer == 512:
                    for t in range(0, w*h, 5):
                        if t + 1 < w*h:
                            long_t[t + 1] = long_t[t+1] - long_t[t] / self.n
                        if t + 2 < w*h:
                            long_t[t + 2] = long_t[t+2] - (long_t[t] + long_t[t+1]) / self.n
                        if t + 3 < w*h:
                            long_t[t + 3] = long_t[t+3] - (long_t[t] + long_t[t+1] + long_t[t+2]) / self.n
                        if t + 4 < w*h:
                            long_t[t + 4] = long_t[t+4] - (long_t[t] + long_t[t+1] + long_t[t+2] + long_t[t+3]) / self.n
                sampled_noise[i, j, :, :] = long_t.reshape([w, h])


        return sampled_noise

class VGG_GBN(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG_GBN, self).__init__()
        self.infl_ratio=1;
        self.features = nn.Sequential(
            BinarizeConv2d(3, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1,
                      bias=True),
            # GaussianNoise(0.1),
            DifferentialNoise(layers=128),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            DifferentialNoise(layers=128),
            # GaussianNoise(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            # GaussianNoise(0.1),
            DifferentialNoise(layers=256),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            # GaussianNoise(0.1),
            DifferentialNoise(layers=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            # GaussianNoise(0.1),
            DifferentialNoise(layers=512),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=True),
            # GaussianNoise(0.1),
            DifferentialNoise(layers=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True)

        )
        self.classifier = nn.Sequential(
            BinarizeLinear(512 * 4 * 4, 1024, bias=True),
            # DifferentialNoise(layers=256),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(1024, 1024, bias=True),
            # DifferentialNoise(layers=64),
            nn.BatchNorm1d(1024),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(1024, num_classes, bias=True),
            # DifferentialNoise(layers=64),
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
        size = x.size
        # print(f"Input: {size}")
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


def vgg_gbn(**kwargs):
    num_classes = kwargs.get( 'num_classes', 10)
    return VGG_GBN(num_classes)
