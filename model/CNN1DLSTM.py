import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_layer1 = nn.Sequential(
        nn.Conv1d(128, 64, 3),
        nn.BatchNorm1d(64)
    )

    self.conv_layer2 = nn.Sequential(
        nn.Conv1d(64, 128, 3),
        nn.BatchNorm1d(128)
    )

    self.conv_layer3 = nn.Sequential(
        nn.Conv1d(128, 256, 3),
        nn.BatchNorm1d(256)
    )

    self.LSTM = nn.LSTM(256, 256, batch_first=True)
    self.maxpool = nn.MaxPool1d(2)
    self.fc = nn.Linear(256, 8)
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    x = x.squeeze(1)
    x = self.conv_layer1(x)
    x = self.maxpool(x)
    x = self.conv_layer2(x)
    x = self.maxpool(x)
    x = self.conv_layer3(x)
    x = x.transpose(1, 2)
    x, (h0, c0) = self.LSTM(x)
    x = h0[-1]
    x = self.dropout(self.fc(x))

    return x