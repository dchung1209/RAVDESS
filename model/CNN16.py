import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input, pool_size=(2, 2)):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x

class CNN16(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_layer1 = ConvBlock(3, 64)
    self.conv_layer2 = ConvBlock(64, 128)
    self.conv_layer3 = ConvBlock(128, 256)
    self.conv_layer4 = ConvBlock(256, 512)
    self.conv_layer5 = ConvBlock(512, 1024)
    self.conv_layer6 = ConvBlock(1024, 2048)
    self.globpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc_layer1 = nn.Sequential(
                  nn.Linear(2048, 527),
                  nn.ReLU()
    )
    self.fc_layer2 = nn.Linear(527, 8)
    self.dropout1 = nn.Dropout(0.1)
    self.dropout2 = nn.Dropout(0.3)
  def forward(self, x):
    x = self.dropout1(self.conv_layer1(x))
    x = self.dropout1(self.conv_layer2(x))
    x = self.dropout1(self.conv_layer3(x))
    x = self.dropout1(self.conv_layer4(x))
    x = self.dropout1(self.conv_layer5(x))
    x = self.dropout1(self.conv_layer6(x))
    x = self.globpool(x)
    x = x.view(x.size(0), -1)
    x = self.dropout2(self.fc_layer1(x))
    x = self.fc_layer2(x)
    return x

