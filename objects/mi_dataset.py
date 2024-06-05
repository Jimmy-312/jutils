from . import GeneralMI
from ..utils.processors import RandomCrop
from torch.utils.data import Dataset

import numpy as np
import torch



class MIDataset(Dataset):
  def __init__(
      self,
      mi_data: GeneralMI,
      cfg,
  ):
    super().__init__()
    self.mi_data = mi_data
    self.img_types = set(mi_data.image_keys + mi_data.label_keys)
    self.cfg = cfg

    if self.cfg.random_crop:
      self.random_crop = RandomCrop(self.cfg.random_crop, true_rand=self.cfg.true_rand)

  # region: basic func
    
  def __len__(self):
    return len(self.mi_data)

  def __getitem__(self, item):
    data_dict = {}
    for img_type in self.img_types:
      data_dict[img_type] = self.fetch_data(item, img_type)
    data_dict = self.process_data(data_dict) 
    return data_dict
  
  def _fetch_data(self, item, img_type):
    data = self.mi_data.images[img_type][item]
    return data
    
  def fetch_data(self, item, img_type):
    data = self._fetch_data(item, img_type)
    if data.shape[0] != 1:
      data = np.expand_dims(data, axis=1)
    return data
  
  # endregion: basic func

  # region: processor func

  def process_data(self, data: dict):
    if self.cfg.random_crop:
      values = list(data.values())
      refers = data[self.mi_data.STD_key]
      values = self.random_crop.mult_gen(values, refers)
      data = dict(zip(data.keys(), values))
    return data
  
  # endregion: processor func
  
  def random_split(self, lengths, seed=None):
    from copy import copy
    if sum(lengths) != len(self):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    if seed is None:
        seed = torch.Generator().manual_seed(42)
    indices = torch.randperm(sum(lengths), generator=seed).tolist()
    return [self.__class__(self.mi_data[indices[offset - length:offset]],
                           copy(self.cfg)) 
                           for offset, length in zip(torch._utils._accumulate(lengths), lengths)]
