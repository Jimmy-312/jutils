from . import MIModel
from einops import rearrange, reduce
from random import random
from functools import partial
from collections import namedtuple
from torch.cuda.amp import autocast
from tqdm import tqdm

import math
import torch
import torch.nn.functional as F


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
identity = lambda x: x


class DiffusionMIModel(MIModel):
  def __init__(self, model, cfg, *args, **kwargs):
    super().__init__(model, cfg, *args, **kwargs)
    self.timesteps = cfg.get('timesteps', 1000)
    self.image_size = cfg.img_shape[1:]
    self.channels = cfg.img_shape[0]

    self.objective = cfg.objective
    assert self.objective in ['pred_noise', 'pred_x0']
    
    self.self_condition = cfg.get('self_condition', False)
    self.offset_noise_strength = cfg.get('offset_noise_strength', 0.1)
    self.is_ddim_sampling = cfg.get('ddim', False)

    self.normalize = self.normalize_to_neg_one_to_one if cfg.auto_normalize else identity
    self.unnormalize = self.unnormalize_to_zero_to_one if cfg.auto_normalize else identity

    beta_schedule = cfg.beta_schedule
    if beta_schedule == 'linear':
      beta_schedule_fn = self.linear_beta_schedule
    elif beta_schedule == 'cosine':
      beta_schedule_fn = self.cosine_beta_schedule
    elif beta_schedule == 'sigmoid':
      beta_schedule_fn = self.sigmoid_beta_schedule
    else:
      raise ValueError(f'unknown beta schedule {beta_schedule}')

    schedule_fn_kwargs = cfg.get('schedule_kwargs', {})
    betas = beta_schedule_fn(self.timesteps, **schedule_fn_kwargs)

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

    timesteps, = betas.shape
    self.num_timesteps = int(timesteps)

    self.sampling_timesteps = cfg.get('sampling_timesteps', timesteps)
    assert self.sampling_timesteps <= timesteps
    self.is_ddim_sampling = self.sampling_timesteps < timesteps
    self.ddim_sampling_eta = cfg.get('ddim_eta', 0.)

    get_val = lambda val: val.to(torch.float32)

    self.betas = get_val(betas)
    self.alphas_cumprod = get_val(alphas_cumprod)
    self.alphas_cumprod_prev = get_val(alphas_cumprod_prev)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    self.sqrt_alphas_cumprod = get_val(torch.sqrt(alphas_cumprod))
    self.sqrt_one_minus_alphas_cumprod = get_val(torch.sqrt(1. - alphas_cumprod))
    self.log_one_minus_alphas_cumprod = get_val(torch.log(1. - alphas_cumprod))
    self.sqrt_recip_alphas_cumprod = get_val(torch.sqrt(1. / alphas_cumprod))
    self.sqrt_recipm1_alphas_cumprod = get_val(torch.sqrt(1. / alphas_cumprod - 1))

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
    self.posterior_variance = get_val(posterior_variance)

    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    self.posterior_log_variance_clipped = get_val(torch.log(posterior_variance.clamp(min = 1e-20)))
    self.posterior_mean_coef1 = get_val(betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
    self.posterior_mean_coef2 = get_val((1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    # https://arxiv.org/abs/2303.09556
    snr = alphas_cumprod / (1 - alphas_cumprod)
    maybe_clipped_snr = snr.clone()
    if cfg.get('min_snr_loss_weight', False):
      maybe_clipped_snr.clamp_(max = cfg.min_snr_gamma)
    
    if self.objective == 'pred_noise':
      self.loss_weight = maybe_clipped_snr / snr
    elif self.objective == 'pred_x0':
      self.loss_weight = maybe_clipped_snr
  
  def forward(self, img, *args, **kwargs):
    b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
    assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
    t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

    img = self.normalize(img)
    return self.p_losses(img, t, *args, **kwargs)

  def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
    b, c, h, w = x_start.shape

    noise = noise if noise is not None else torch.randn_like(x_start)

    # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

    offset_noise_strength = offset_noise_strength if offset_noise_strength \
      else self.offset_noise_strength

    if offset_noise_strength > 0.:
      offset_noise = torch.randn(x_start.shape[:2], device = self.device)
      noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

    # noise sample

    x = self.q_sample(x_start = x_start, t = t, noise = noise)

    # if doing self-conditioning, 50% of the time, predict x_start from current set of times
    # and condition with unet with that
    # this technique will slow down training by 25%, but seems to lower FID significantly

    x_self_cond = None
    if self.self_condition and random() < 0.5:
      with torch.no_grad():
        x_self_cond = self.model_predictions(x, t).pred_x_start
        x_self_cond.detach_()

    # predict and take gradient step

    model_out = self.model(x, t, x_self_cond)

    if self.objective == 'pred_noise':
      target = noise
    elif self.objective == 'pred_x0':
      target = x_start
    else:
      raise ValueError(f'unknown objective {self.objective}')

    loss = F.mse_loss(model_out, target, reduction = 'none')
    loss = reduce(loss, 'b ... -> b', 'mean')

    loss = loss * self.extract(self.loss_weight, t, loss.shape)
    return loss.mean()

  def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
    model_output = self.model(x, t, x_self_cond)
    maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

    if self.objective == 'pred_noise':
      pred_noise = model_output
      x_start = self.predict_start_from_noise(x, t, pred_noise)
      x_start = maybe_clip(x_start)

      if clip_x_start and rederive_pred_noise:
        pred_noise = self.predict_noise_from_start(x, t, x_start)

    elif self.objective == 'pred_x0':
      x_start = model_output
      x_start = maybe_clip(x_start)
      pred_noise = self.predict_noise_from_start(x, t, x_start)

    return ModelPrediction(pred_noise, x_start)

  def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
    preds = self.model_predictions(x, t, x_self_cond)
    x_start = preds.pred_x_start

    if clip_denoised:
      x_start.clamp_(-1., 1.)

    model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
    return model_mean, posterior_variance, posterior_log_variance, x_start
  
  @autocast(enabled = False)
  def q_sample(self, x_start, t, noise = None):
    noise = noise if noise is not None else torch.randn_like(x_start)

    return (
      self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
      self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )
  
  @torch.inference_mode()
  def sample(self, batch_size = 16, return_all_timesteps = False):
    (h, w), channels = self.image_size, self.channels
    sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
    return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)

  @torch.inference_mode()
  def ddim_sample(self, shape, return_all_timesteps = False):
    batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

    times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    img = torch.randn(shape, device = device)
    imgs = [img]

    x_start = None

    for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
      time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
      self_cond = x_start if self.self_condition else None
      pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

      if time_next < 0:
        img = x_start
        imgs.append(img)
        continue

      alpha = self.alphas_cumprod[time]
      alpha_next = self.alphas_cumprod[time_next]

      sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
      c = (1 - alpha_next - sigma ** 2).sqrt()

      noise = torch.randn_like(img)

      img = x_start * alpha_next.sqrt() + \
        c * pred_noise + \
        sigma * noise

      imgs.append(img)

    ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    ret = self.unnormalize(ret)
    return ret

  @torch.inference_mode()
  def p_sample_loop(self, shape, return_all_timesteps = False):
    batch, device = shape[0], self.device

    img = torch.randn(shape, device = device)
    imgs = [img]

    x_start = None

    for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
      self_cond = x_start if self.self_condition else None
      img, x_start = self.p_sample(img, t, self_cond)
      imgs.append(img)

    ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    ret = self.unnormalize(ret)
    return ret
  
  @torch.inference_mode()
  def p_sample(self, x, t: int, x_self_cond = None):
    b, *_, device = *x.shape, self.device
    batched_times = torch.full((b,), t, device = device, dtype = torch.long)
    model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
    noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
    pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
    return pred_img, x_start

  def q_posterior(self, x_start, x_t, t):
    posterior_mean = (
      self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
      self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
    return posterior_mean, posterior_variance, posterior_log_variance_clipped

  def predict_start_from_noise(self, x_t, t, noise):
    return (
      self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
      self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )

  def predict_noise_from_start(self, x_t, t, x0):
    return (
      (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
      self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    )
  
  def extract(self, a, t, x_shape):
    a = a.to(self.device)
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
  
  @staticmethod
  def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

  @staticmethod
  def unnormalize_to_zero_to_one(t):
      return (t + 1) * 0.5
  
  @staticmethod
  def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

  @staticmethod
  def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

  @staticmethod
  def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)