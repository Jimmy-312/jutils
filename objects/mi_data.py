from scipy import rand
from torch import _test_serialization_subcmul
from jutils.objects import GeneralMI, MIDataset
from torch.utils.data import DataLoader, random_split
from omegaconf import DictConfig
from torchvision import transforms

import pytorch_lightning as pl
import torch



class MIData(pl.LightningDataModule):
  def __init__(self, cfg: DictConfig):
    super().__init__()
    self.data_path = cfg.data_path
    self.data_shape = cfg.data_shape
    self.data_set = cfg.data_set

    self.batch_size = cfg.batch_size
    self.test_batch_size = cfg.test_batch_size

    self.transform = transforms.Compose([
            transforms.ToTensor(),
            ])


  def prepare_data(self):
    process_param = {
      'ct_window': None,
      'norm': 'PET',  # only min-max,
      'shape': self.data_shape,  # [320, 320, 240]
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
    self.mi_data = GeneralMI.init_data(self.data_set, self.data_path, 
                                       img_type, process_param)
    self.mi_data.pre_load(threads=48)
    self.dataset = MIDataset(self.mi_data, self.transform)

  def setup(self, stage=None):
    train_ratio = 0.8
    test_radio = 0.1
    train_size = int(train_ratio * len(self.dataset))
    test_size = int(test_radio * len(self.dataset))
    val_size = len(self.dataset) - train_size - test_size

    self.mi_train, self.mi_val, self.mi_test = random_split(
      self.dataset,
      [train_size, val_size, test_size],
      torch.manual_seed(0)
      )
      
  def train_dataloader(self):
    return DataLoader(self.mi_train, batch_size=self.batch_size, 
                      shuffle=True, num_workers=32)

  def val_dataloader(self):
    return DataLoader(self.mi_val, batch_size=self.batch_size, 
                      shuffle=False, num_workers=32)

  def test_dataloader(self):
    return DataLoader(self.mi_test, batch_size=self.test_batch_size, 
                      shuffle=False, num_workers=32)