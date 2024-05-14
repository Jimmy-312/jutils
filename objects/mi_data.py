from . import GeneralMI, MIDataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

import pytorch_lightning as pl
import torch



class MIData(pl.LightningDataModule):
  def __init__(self, cfg: DictConfig):
    super().__init__()
    self.data_path = cfg.data_path
    self.data_shape = cfg.mi_config.process_param.shape
    self.data_set = cfg.data_set
    self.seed = torch.Generator().manual_seed(cfg.seed)

    self.batch_size = cfg.batch_size
    self.val_batch_size = cfg.val_batch_size
    self.test_batch_size = cfg.test_batch_size

    process_param = OmegaConf.to_container(cfg.mi_config.process_param, 
                                           resolve=True)
    img_type = OmegaConf.to_container(cfg.mi_config.img_type, resolve=True)
    self.mi_data = GeneralMI.init_data(self.data_set, self.data_path, 
                                       img_type, process_param)
    
    self.transform = cfg.transform

  def prepare_data(self):
    self.mi_data.pre_load(threads=48)
    self.dataset = MIDataset(self.mi_data, self.transform)

  def setup(self, stage=None):
    # fit validate test predict

    train_ratio = 0.8
    test_radio = 0.1
    train_size = int(train_ratio * len(self.dataset))
    test_size = int(test_radio * len(self.dataset))
    val_size = len(self.dataset) - train_size - test_size

    self.mi_train, self.mi_val, self.mi_test = self.dataset.random_split(
      [train_size, val_size, test_size],
      self.seed
      )
    self.mi_val.cfg.random_crop = None
    self.mi_test.cfg.random_crop = None
      
  def train_dataloader(self):
    return DataLoader(self.mi_train, batch_size=self.batch_size, 
                      shuffle=True, num_workers=32, pin_memory=True)

  def val_dataloader(self):
    return DataLoader(self.mi_val, batch_size=self.val_batch_size, 
                      shuffle=False, num_workers=32, pin_memory=True)

  def test_dataloader(self):
    return DataLoader(self.mi_test, batch_size=self.test_batch_size, 
                      shuffle=False, num_workers=32, pin_memory=True)