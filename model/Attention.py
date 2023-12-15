import torch
import torch.nn as nn

class Attention(nn.Module):
  def __init__(self):
    super().__init__()
    self.dense = nn.Linear(in_features = 64, out_features = 64)
    self.lamb_val = 0.3
    self.u = nn.Parameter(torch.rand(1, 64), requires_grad=True)
    # nn.init.kaiming_uniform_(self.u, a=0, mode='fan_in', nonlinearity='leaky_relu')
    self.tanh = nn.Tanh()

  def forward(self, x):
    batch_size, channel_size, variable_length = x.size()
    # N x C X L -> N x L x C
    x = torch.transpose(x, -1, -2)
    a = self.tanh(self.dense(x))
    a = self.u @ a.transpose(-1,-2) * self.lamb_val
    a = torch.softmax(a, dim=-1)
    out = a @ x
    out = out.transpose(-1, -2)
    output = torch.sum(out, dim=2)
    return output
  