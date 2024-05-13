from unicodedata import numeric
from torch.utils.data import Dataset
from jutils.objects import GeneralMI
from jutils.utils.img_process import get_random_window, get_sample

import numpy as np



class MIDataset(Dataset):
  def __init__(
      self,
      mi_data: GeneralMI,
      transforms=None,
      random_crop=None,
  ):
    super().__init__()
    self.mi_data = mi_data
    self.img_types = set(mi_data.image_keys + mi_data.label_keys)
    self.transforms = transforms
    self.random_crop = random_crop

  def __len__(self):
    return len(self.mi_data)

  def __getitem__(self, item):
    data_dict = {}
    pos = self.pre_process(item)
    for img_type in self.img_types:
      data_dict[img_type] = self.fetch_data(item, img_type, pos)
    return data_dict
  
  def _fetch_data(self, item, img_type):
    data = self.mi_data.images[img_type][item]
    if self.transforms:
      data = self.transforms(data)
    return data
    
  def fetch_data(self, item, img_type, pos=None):
    data = self._fetch_data(item, img_type)
    data = self.process_data(data, pos)    
    if data.shape[0] != 1:
      data = np.expand_dims(data, axis=1)
    return data
  
  def pre_process(self, item):
    pos = None
    if self.random_crop and isinstance(item, int):
      pos = get_random_window(self.mi_data.images[self.mi_data.STD_key][item], 
                              self.random_crop)
    return pos
  
  def process_data(self, data, pos=None):
    if self.random_crop:
      data = get_sample(data, pos, self.random_crop)
    return data





if __name__ == '__main__':
  testset = ['30G', '240G']
  datacsv = '/z3/home/xai_test/xai-omics/data/02-RLD/rld_data.csv'

  mi_data = GeneralMI.get_test_sample(datacsv, testset)

  # mi_data.pre_load(threads=48)
  dataset = MIDataset(
    mi_data, 
    random_crop=(64, 64, 64),
  )
  from torch.utils.data import DataLoader
  loader = DataLoader(
      dataset,
      batch_size=4,
      shuffle=True,
      num_workers=32,
      drop_last=True
  )

  for batch in loader:
    print(batch['30G'].shape)
    break
  import matplotlib.pyplot as plt
  print(dataset[2]['30G'].shape)
  plt.imshow(dataset[0]['30G'][0, 32], cmap='gist_yarg')
  plt.savefig('test.png')
