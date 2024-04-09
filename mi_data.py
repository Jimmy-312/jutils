from torch.utils.data import DataLoader, Dataset
from general_mi import GeneralMI
from img_process import gen_windows

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
                      image_keys=data_set,
                      label_keys=[],
                      pid=pid, process_param=process_param, img_type=img_type)
  total_data = len(mi_data)
  mi_data.rm_void_data()
  print(f"Loaded data: {len(mi_data)}/{total_data}")

  return mi_data




if __name__ == '__main__':
  testset = ['30G', '240G']
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

  dataset = MIDataset(
    mi_data, 
    crop_window=(64, 64),
    flatten3d=True
  )
  loader = DataLoader(
      dataset,
      batch_size=4,
      shuffle=True,
      num_workers=32,
      drop_last=True
  )

  for i, data in enumerate(loader):
    # print(data, i)
    break

  import matplotlib as mpl

  mpl.use('Agg')
  import matplotlib.pyplot as plt

  print(data['240G'].shape)
  # plt.imshow(data['30G'][0, 0, 32])
  # plt.savefig('test.png')
