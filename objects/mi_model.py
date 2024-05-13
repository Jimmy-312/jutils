import pytorch_lightning as pl
import torch



class MIModel(pl.LightningModule):
  def __init__(self, model, cfg, *args, **kwargs):
    super().__init__()
    self.model = model
    self.lr = cfg.lr

  def forward(self, x):
    return self.model(x)
  
  def training_step(self, batch, batch_idx):
    pass

  def validation_step(self, batch, batch_idx):
    pass

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer
