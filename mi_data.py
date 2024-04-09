from torch.utils.data import DataLoader, Dataset
from .general_mi import GeneralMI
from torch import Tensor

import numpy as np



class MIDataset(Dataset):
  def __init__(
      self,
      mi_data: GeneralMI,

  ):
    super().__init__()
    self.mi_data = mi_data

  def __len__(self):
    return len(self.mi_data)

  def __getitem__(self, item):
    features = np.expand_dims(self.mi_data.images['30G'][item], axis=0)
    targets = np.expand_dims(self.mi_data.labels['240G'][item], axis=0)
    return targets, {}


def init_data(data_set, csv_path, img_type, process_param):
  data = np.genfromtxt(csv_path, delimiter=',', dtype=str)
  types = data[0][1:]
  pid = data[1:, 0]
  path_array = data[1:, 1:]

  img_dict = {}
  for i, type_name in enumerate(types):
    img_path = path_array[:, i]
    img_dict[type_name] = {'path': img_path}

  mi_data = GeneralMI(img_dict,
                      image_keys=data_set[0],
                      label_keys=data_set[1],
                      pid=pid, process_param=process_param, img_type=img_type)
  total_data = len(mi_data)
  mi_data.rm_void_data()
  print(f"Loaded data: {len(mi_data)}/{total_data}")

  return mi_data


if __name__ == '__main__':
  testset = (['30G'], ['240G'])
  datacsv = '/z3/home/xai_test/xai-omics/data/02-RLD/rld_data.csv'
  process_param = {
    'ct_window': None,
    'norm': 'PET',  # only min-max,
    'shape': [440, 440, 560],  # [320, 320, 240]
    'crop': [0, 0, 5],  # [30, 30, 10]
    'clip': None,  # [1, None]
  }

  img_type = {
    'CT': ['CT'],
    'PET': ['30G', '20S', '40S',
            '60G-1', '60G-2', '60G-3',
            '90G', '120S', '120G', '240G', '240S'],
    'MASK': ['CT_seg'],
    'STD': ['30G'],
  }

  mi_data = init_data(testset, datacsv, img_type, process_param)

  import matplotlib as mpl

  mpl.use('Agg')
  import matplotlib.pyplot as plt

  print(mi_data.images_raw['30G'][0].shape)
  plt.imshow(mi_data.images['30G'][0][240, :, :])
  plt.savefig('test.png')
