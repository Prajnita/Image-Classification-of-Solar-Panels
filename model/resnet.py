from torch.utils.data import Dataset
import torch
import torch.nn as nn
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision as tv
import imageio

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: stimmt das weil wir RGB Bilder haben?
        self.input_size = 3
        self.conv1 = nn.Conv2d(self.input_size, 64, 7, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.resblock1 = ResBlock(64, 64, 1)
        self.resblock2 = ResBlock(64, 128, 2)
        self.resblock3 = ResBlock(128, 256, 2)
        self.resblock4 = ResBlock(256, 512, 2)
        self.globalavgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten1 = nn.Flatten()
        self.fcl1 = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.resblock1.forward(x)
        x = self.resblock2.forward(x)
        x = self.resblock3.forward(x)
        x = self.resblock4.forward(x)
        x = self.globalavgpool1(x)
        x = self.flatten1(x)
        #print(x.size())
        x = self.fcl1(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.res_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.res_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride)


    def forward(self, x):
        residual = self.conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x + residual