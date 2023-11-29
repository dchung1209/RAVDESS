from torch.utils.data import Dataset, DataLoader, RandomSampler
import torchvision.transforms as transforms
import torch
import os
import numpy as np
from utilities import RAVDESS, AudioUtils

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


class DataModule(AudioUtils, RAVDESS):
    def __init__(self, root, channel_num, transform):
      super().__init__(root)
      self.root = root
      self.channel_num = channel_num
      self.transform = transform
      self.df = self.load(root)
      self.invert_emotions = dict(zip(self.emotions.values(), self.emotions.keys()))

    def transform_dataframe(self):
      self.df['Path'] = self.df['Path'].apply(lambda x: self.MelSpec(x, self.channel_num))
      self.df['Emotion'] = self.df['Emotion'].apply(lambda x: int(self.invert_emotions[x]) - 1)
      self.df['ID'] = self.df['ID'].apply(lambda x: int(x))
    
    def split_dataframe(self, actor_id = []):
      return self.df[self.df['ID'].isin(actor_id)]
    
    def get_dataloader(self, x, y):
      dataset = RAVDESS_Dataset(x, y, self.transform, True)
      sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(41))
      dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
      return dataloader

    def get_si_dataloader(self, train_id, test_id):
      train_df = self.split_dataframe(train_id)
      test_df = self.split_dataframe(test_id)
      return self.get_dataloader(train_df['Path'], train_df["Emotion"]), self.get_dataloader(test_df['Path'], test_df["Emotion"])
    
    def get_dataframe(self):
      return self.df


      