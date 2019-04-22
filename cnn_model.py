from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, n_channels, n_classes):
        """
        Initializes CNN object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.maxpool5 = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(1, stride=1, padding=0)
        self.linear = nn.Linear(512, 10)
        print("init finish")

    def forward(self, x):
        """
        Performs forward pass of the input.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        # print(x.size())
        x = self.conv1(x)
        # print("conv1", x.size())
        x = self.bn1(x)
        # print("bn1", x.size())
        x = self.maxpool1(F.relu(x))
        # print("maxpool1", x.size())
        x = self.conv2(x)
        # print("conv2", x.size())
        x = self.bn2(x)
        # print("bn2", x.size())
        x = self.maxpool2(F.relu(x))
        # print("maxpool2", x.size())
        x = self.conv3(x)
        # print("conv3", x.size())
        x = self.bn3(x)
        # print("bn3", x.size())
        x = self.conv4(F.relu(x))
        # print("conv4", x.size())
        x = self.bn4(x)
        # print("bn4", x.size())
        x = self.maxpool3(F.relu(x))
        # print("maxpool3", x.size())
        x = self.conv5(x)
        # print("conv5", x.size())
        x = self.bn5(x)
        # print("bn5", x.size())
        x = self.conv6(F.relu(x))
        # print("conv6", x.size())
        x = self.bn6(x)
        # print("bn6", x.size())
        x = self.maxpool4(F.relu(x))
        # print("maxpool4", x.size())
        x = self.conv7(x)
        # print("conv7", x.size())
        x = self.bn7(x)
        # print("bn7", x.size())
        x = self.conv8(F.relu(x))
        # print("conv8", x.size())
        x = self.bn8(x)
        # print("bn8", x.size())
        x = self.maxpool5(F.relu(x))
        # print("maxpool5", x.size())
        x = self.avgpool(x)
        # print("avgpool", x.size())
        x = torch.squeeze(x)
        # print("squeeze", x.size())
        x = self.linear(x)
        # print("linear", x.size())
        return x
