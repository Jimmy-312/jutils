import torch
import torch.nn as nn

from lightning import LightningModule



class MIModel(LightningModule):
  # Abstract Class
  
  def __init__(self, model, cfg, *args, **kwargs):
    super().__init__()
    self.model = model
    self.lr = cfg.lr

  def forward(self, x):
    return self.model(x)
  
  def training_step(self, batch, batch_idx, *args, **kwargs):
    pass

  def validation_step(self, batch, batch_idx):
    pass

  def test_step(self, batch, batch_idx):
    pass

  def on_train_epoch_end(self):
    pass

  def on_validation_epoch_end(self):
    pass

  def training_step_end(self, outputs):
    pass

  def configure_optimizers(self):
    if hasattr(self, 'custom_optimizers'):
      return self.custom_optimizers()
    else:
      optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
      return optimizer


class GANMIModel(MIModel):
  def __init__(self, generator, discriminator, cfg, *args, **kwargs):
    super().__init__(generator, cfg, *args, **kwargs)
    self.automatic_optimization = False
    self.adversarial_loss = nn.BCELoss()
    
    if cfg.get('d_lr'):
      self.d_lr = cfg.d_lr
    else: 
      self.d_lr = self.lr

    self.discriminator = discriminator
    self.z_dim = cfg.z_dim

    self.g_acc_bs = cfg.g_acc
    self.d_acc_bs = cfg.d_acc
  
  def generate(self, *args):
    return self.model(*args)
  
  def random_z(self, n):
    return torch.randn((n, self.z_dim), device=self.device)
  
  def custom_optimizers(self):
    betas = (0.5, 0.999)
    opt_g = torch.optim.Adam(self.model.parameters(), 
                             lr=self.lr, betas=betas)
    opt_d = torch.optim.Adam(self.discriminator.parameters(), 
                             lr=self.d_lr, betas=betas)
    return opt_g, opt_d

