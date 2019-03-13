import torch
import torch.nn.functional as F
from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def downsample(inplanes, outplanes, stride):
    return nn.Sequential(
        conv1x1(inplanes, outplanes, stride),
        # nn.BatchNorm2d(outplanes)
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, use_conv=True):
        super(ResidualBlock, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        if use_conv:
            # self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = conv3x3(in_planes, out_planes)
        else:
            # self.bn1 = nn.BatchNorm1d(in_planes)
            self.conv1 = nn.Linear(in_planes, out_planes)

        self.droprate = dropRate

    def forward(self, x):
        # out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv1(self.relu(x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return x + out


class DenseBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, use_conv=True):
        super(DenseBlock, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        if use_conv:
            # self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = conv3x3(in_planes, out_planes)
        else:
            # self.bn1 = nn.BatchNorm1d(in_planes)
            self.conv1 = nn.Linear(in_planes, out_planes)

        self.droprate = dropRate

    def forward(self, x):
        # out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv1(self.relu(x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class LinearBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, use_conv=True):
        super(LinearBlock, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        if use_conv:
            # self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = conv3x3(in_planes, out_planes)
        else:
            # self.bn1 = nn.BatchNorm1d(in_planes)
            self.conv1 = nn.Linear(in_planes, out_planes)

        self.droprate = dropRate

    def forward(self, x):
        # out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv1(self.relu(x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return out


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class YourModel(nn.Module):
    def __init__(self, ):
        super(YourModel, self).__init__()

    def forward(self, x):
        pass

    def generate_function(self, x):
        pass
