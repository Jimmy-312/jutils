from torch.utils.data import Dataset
from objects import GeneralMI
from utils.img_process import gen_windows

import numpy as np



class MIDataset(Dataset):
  def __init__(
      self,
      mi_data: GeneralMI,
      crop_window: tuple = None,
      flatten3d: bool = False
  ):
    super().__init__()
    self.mi_data = mi_data
    self.img_types = set(mi_data.image_keys + mi_data.label_keys)
    self.flatten3d = flatten3d
    self.crop_window = crop_window

  def __len__(self):
    return len(self.mi_data)

  def __getitem__(self, item):
    data_dict = {}
    for img_type in self.img_types:
      data_dict[img_type] = self._fetch_data(item, img_type)
    return data_dict
  
  def _fetch_data(self, item, img_type):
    data: np.ndarray = self.mi_data.images[img_type][item]
    if self.flatten3d:
      idx = np.random.randint(0, data.shape[0])
      data = data[idx]
    if self.crop_window is not None:
      data = gen_windows(data, self.crop_window)
    return np.expand_dims(data, axis=0)





if __name__ == '__main__':
  testset = ['30G', '240G']
  datacsv = '/z3/home/xai_test/xai-omics/data/02-RLD/rld_data.csv'

  mi_data = GeneralMI.get_test_sample(datacsv, testset)

  # mi_data.pre_load(threads=48)
  # dataset = MIDataset(
  #   mi_data, 
  #   crop_window=(64, 64),
  #   flatten3d=True
  # )
  # loader = DataLoader(
  #     dataset,
  #     batch_size=4,
  #     shuffle=True,
  #     num_workers=32,
  #     drop_last=True
  # )

  import matplotlib.pyplot as plt
  
  plt.imshow(mi_data.images['30G'][0][200], cmap='gist_yarg')
  plt.savefig('test.png')
