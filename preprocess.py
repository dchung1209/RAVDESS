from torch.utils.data import Dataset, DataLoader, RandomSampler
import torchvision.transforms as transforms
import torch
import os
from utilities import RAVDESS, ToMelSpec

class RAVDESS_Dataset(Dataset):
  def __init__(self, x, y, transform, train = True):
    self.emotion = y.to_list()
    self.path = x.to_list()
    self.transform = transform

  def __len__(self):
    return len(self.emotion)

  def __getitem__(self, idx):
    spec = self.transform(np.uint8(self.path[idx]))
    y = self.emotion[idx]
    return spec, y


class DataModule():
    def __init__(self, root, channel_num, transform):
      self.root = root
      self.channel_num = channel_num
      self.transform = transform
      self.dataloader = RAVDESS(self.root)
    
    def mel_transform(self):
      Mels = ToMelSpec.MelSpec(self.dataloader)
      if (self.channel_num == 1):
        Mels['Path'] = Mels['Path'].apply(lambda x: x[..., 0])
      return Mels
    
    def split_dataloader(self, actor_id = []):
      return self.dataloader[self.dataloader['ID'].isin(actor_id)]  
    
    def get_dataloader(self):
      dataset = RAVDESS_Dataset(x, y, self.transform, True)
      sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(41))
      dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
      return dataloader
