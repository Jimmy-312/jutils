from collections import namedtuple
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial, wraps
from packaging import version
from torch import einsum, nn

import math
import torch
import torch.nn.functional as F


AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

def cast_tuple(t, length = 1):
  if isinstance(t, tuple):
    return t
  return ((t,) * length)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)


def Upsample(dim, dim_out = None):
  return nn.Sequential(
    nn.Upsample(scale_factor = 2, mode = 'nearest'),
    nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
  )

def Downsample(dim, dim_out = None):
  return nn.Sequential(
    Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
    nn.Conv2d(dim * 4, default(dim_out, dim), 1)
  )

class RMSNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.scale = dim ** 0.5
    self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

  def forward(self, x):
    return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
  def __init__(self, dim, theta = 10000):
    super().__init__()
    self.dim = dim
    self.theta = theta

  def forward(self, x):
    device = x.device
    half_dim = self.dim // 2
    emb = math.log(self.theta) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = x[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
  """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
  """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

  def __init__(self, dim, is_random = False):
    super().__init__()
    assert (dim % 2) == 0
    half_dim = dim // 2
    self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

  def forward(self, x):
    x = rearrange(x, 'b -> b 1')
    freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
    fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
    fouriered = torch.cat((x, fouriered), dim = -1)
    return fouriered

# building block modules

class Block(nn.Module):
  def __init__(self, dim, dim_out):
    super().__init__()
    self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
    self.norm = RMSNorm(dim_out)
    self.act = nn.SiLU()

  def forward(self, x, scale_shift = None):
    x = self.proj(x)
    x = self.norm(x)

    if scale_shift is not None:
      scale, shift = scale_shift
      x = x * (scale + 1) + shift

    x = self.act(x)
    return x

class ResnetBlock(nn.Module):
  def __init__(self, dim, dim_out, *, time_emb_dim = None):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.SiLU(),
      nn.Linear(time_emb_dim, dim_out * 2)
    ) if time_emb_dim is not None else None

    self.block1 = Block(dim, dim_out)
    self.block2 = Block(dim_out, dim_out)
    self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

  def forward(self, x, time_emb = None):

    scale_shift = None
    if self.mlp is not None and time_emb is not None:
      time_emb = self.mlp(time_emb)
      time_emb = rearrange(time_emb, 'b c -> b c 1 1')
      scale_shift = time_emb.chunk(2, dim = 1)

    h = self.block1(x, scale_shift = scale_shift)

    h = self.block2(h)

    return h + self.res_conv(x)

class LinearAttention(nn.Module):
  def __init__(
    self,
    dim,
    heads = 4,
    dim_head = 32,
    num_mem_kv = 4
  ):
    super().__init__()
    self.scale = dim_head ** -0.5
    self.heads = heads
    hidden_dim = dim_head * heads

    self.norm = RMSNorm(dim)

    self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
    self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

    self.to_out = nn.Sequential(
      nn.Conv2d(hidden_dim, dim, 1),
      RMSNorm(dim)
    )

  def forward(self, x):
    b, c, h, w = x.shape

    x = self.norm(x)

    qkv = self.to_qkv(x).chunk(3, dim = 1)
    q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

    mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
    k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

    q = q.softmax(dim = -2)
    k = k.softmax(dim = -1)

    q = q * self.scale

    context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

    out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
    out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
    return self.to_out(out)

class Attention(nn.Module):
  def __init__(
    self,
    dim,
    heads = 4,
    dim_head = 32,
    num_mem_kv = 4,
    flash = False
  ):
    super().__init__()
    self.heads = heads
    hidden_dim = dim_head * heads

    self.norm = RMSNorm(dim)
    self.attend = Attend(flash = flash)

    self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
    self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
    self.to_out = nn.Conv2d(hidden_dim, dim, 1)

  def forward(self, x):
    b, c, h, w = x.shape

    x = self.norm(x)

    qkv = self.to_qkv(x).chunk(3, dim = 1)
    q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

    mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
    k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

    out = self.attend(q, k, v)

    out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
    return self.to_out(out)
  

class Attend(nn.Module):
  def __init__(
    self,
    dropout = 0.,
    flash = False,
    scale = None
  ):
    super().__init__()
    self.dropout = dropout
    self.scale = scale
    self.attn_dropout = nn.Dropout(dropout)

    self.flash = flash
    assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

    # determine efficient attention configs for cuda and cpu

    self.cpu_config = AttentionConfig(True, True, True)
    self.cuda_config = None

    if not torch.cuda.is_available() or not flash:
      return

    device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

    if device_properties.major == 8 and device_properties.minor == 0:
      print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
      self.cuda_config = AttentionConfig(True, False, False)
    else:
      print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
      self.cuda_config = AttentionConfig(False, True, True)

  def flash_attn(self, q, k, v):
    _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

    if exists(self.scale):
      default_scale = q.shape[-1]
      q = q * (self.scale / default_scale)

    q, k, v = map(lambda t: t.contiguous(), (q, k, v))

    # Check if there is a compatible device for flash attention

    config = self.cuda_config if is_cuda else self.cpu_config

    # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

    with torch.backends.cuda.sdp_kernel(**config._asdict()):
      out = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p = self.dropout if self.training else 0.
      )

      return out

  def forward(self, q, k, v):
    """
    einstein notation
    b - batch
    h - heads
    n, i, j - sequence length (base sequence length, source, target)
    d - feature dimension
    """

    q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

    if self.flash:
      return self.flash_attn(q, k, v)

    scale = default(self.scale, q.shape[-1] ** -0.5)

    # similarity

    sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

    # attention

    attn = sim.softmax(dim = -1)
    attn = self.attn_dropout(attn)

    # aggregate values

    out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

    return out