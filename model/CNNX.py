import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNX(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3)),
        nn.ReLU()
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(8, 16, (3, 3)),
        nn.ReLU()
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(16, 32, (3, 3)),
        nn.ReLU()
    )

    self.fc1 = nn.Sequential(
        nn.Linear(13440, 2048),
        nn.ReLU()
    )

    self.fc2 = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.ReLU()
    )
    self.fc3 = nn.Linear(2048, 8)
    self.avg_pool = nn.AvgPool2d((2, 2))

  def forward(self, x):
    x = self.avg_pool(self.conv1(x))
    x = self.avg_pool(self.conv2(x))
    x = self.avg_pool(self.conv3(x))
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x
