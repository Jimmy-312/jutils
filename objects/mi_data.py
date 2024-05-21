from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig

from lightning import LightningDataModule
import torch


class MIData(LightningDataModule):
  # Abstract Class

  def __init__(self, cfg: DictConfig):
    super().__init__()
    
    self.seed = torch.Generator().manual_seed(cfg.seed)
    self.batch_size = cfg.batch_size
    self.val_batch_size = cfg.val_batch_size
    self.test_batch_size = cfg.test_batch_size
    self.loader = cfg.loader
    self.data_split = cfg.data_split

  def prepare_data(self):
    # data preloader
    self.dataset = Dataset()

  def setup(self, stage=None):
    # fit validate test predict
    train_size = int(self.data_split[0] * len(self.dataset))
    test_size = int(self.data_split[1] * len(self.dataset))
    val_size = len(self.dataset) - train_size - test_size

    self.train = Dataset()
    self.val = Dataset()
    self.test = Dataset()

    return train_size, test_size, val_size
      
  def train_dataloader(self):
    return DataLoader(self.train, batch_size=self.batch_size, 
                      shuffle=self.loader.shuffle, 
                      num_workers=self.loader.num_workers, 
                      pin_memory=self.loader.pin_memory)

  def val_dataloader(self):
    return DataLoader(self.val, batch_size=self.val_batch_size, 
                      shuffle=False, 
                      num_workers=self.loader.num_workers, 
                      pin_memory=self.loader.pin_memory)

  def test_dataloader(self):
    return DataLoader(self.test, batch_size=self.test_batch_size, 
                      shuffle=False, 
                      num_workers=self.loader.num_workers, 
                      pin_memory=self.loader.pin_memory)