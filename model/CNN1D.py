import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_layer1 = nn.Sequential(
        nn.Conv1d(128, 256, 3),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1)
    )

    self.conv_layer2 = nn.Sequential(
        nn.Conv1d(256, 256, 3),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1)
    )

    self.conv_layer3 = nn.Sequential(
        nn.Conv1d(256, 512, 5),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2)
    )


    self.maxpool = nn.MaxPool1d(2)
    self.dropout = nn.Dropout(p=0.5)

    self.fc_layer1 = nn.Linear(31232, 512)
    self.fc_layer2 = nn.Linear(512, 8)

  def forward(self, x):
    x = x.squeeze(1)
    x = self.conv_layer1(x)
    x = self.conv_layer2(x)
    x = self.maxpool(x)
    x = self.conv_layer3(x)
    x = self.maxpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc_layer1(x)
    x = self.fc_layer2(x)
    return x
  