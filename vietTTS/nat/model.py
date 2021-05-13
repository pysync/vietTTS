import math
from typing import NamedTuple, Optional, Tuple

import einops
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from einops.einops import rearrange
from jax.numpy import ndarray
from vietTTS.tacotron.config import FLAGS

from .config import FLAGS, AcousticInput, DurationInput


def get_lightconv_fn(kernel_size, num_heads):
  def f(x): return hk.Conv1D(num_heads, kernel_size, 1, 1, 'SAME', False, feature_group_count=num_heads)(x)
  return lambda w, x: hk.without_apply_rng(hk.transform(f)).apply({'conv1_d': {'w': w}}, x)


def posenc(x):
  B, L, D = x.shape
  pos = jnp.arange(0, L)[:, None]
  div_term = jnp.exp(
      jnp.arange(0, D, 2, dtype=jnp.float32)[None, :]
      * (-math.log(10_000) / D)
  )
  x1 = jnp.sin(pos * div_term)
  x2 = jnp.cos(pos * div_term)
  x_ = jnp.concatenate((x1, x2), axis=-1)
  return x + x_[None]


class LightConv(hk.Module):
  def __init__(self, dim, kernel_size, num_heads=8, dropout_rate=0.1, is_training=True):
    super().__init__()
    self.kernel_size = kernel_size
    self.num_heads = num_heads
    self.dropout_rate = dropout_rate
    self.is_training = is_training
    self.conv = get_lightconv_fn(kernel_size, num_heads)

  def __call__(self, x):
    B, W, C = x.shape
    R = C // self.num_heads
    x = einops.rearrange(x, 'B W (H R) -> B W H R', H=self.num_heads)
    w_shape = (self.kernel_size, 1, self.num_heads)
    fan_in_shape = np.prod(w_shape[:-1])
    stddev = 1. / np.sqrt(fan_in_shape)
    w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = w = hk.get_parameter("w", w_shape, x.dtype, init=w_init)
    w = jax.nn.softmax(w, axis=0)
    if self.is_training:
      w = hk.dropout(hk.next_rng_key(), self.dropout_rate, w)
    x = hk.vmap(self.conv, in_axes=(None, 3), out_axes=3)(w, x)
    x = einops.rearrange(x, 'B W H R -> B W (H R)', H=self.num_heads)
    return x


class LConvBlock(hk.Module):
  def __init__(self, dim, kernel_size, num_heads=8, dropout_rate=0.1, is_training=True):
    super().__init__()

    self.layernorm1 = hk.LayerNorm(-1, True, True)
    self.glu_fc = hk.Linear(dim*2)
    self.lconv = LightConv(dim, kernel_size, num_heads, dropout_rate, is_training)

    self.layernorm2 = hk.LayerNorm(-1, True, True)
    self.ff_fc1 = hk.Linear(dim*4)
    self.ff_fc2 = hk.Linear(dim)

  def __call__(self, x):
    x_res = x
    x = self.layernorm1(x)
    x = self.glu_fc(x)
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = x1 * jax.nn.sigmoid(x2)
    x = self.lconv(x)
    x = x + x_res
    x_res = x
    x = self.layernorm2(x)
    x = self.ff_fc1(x)
    x = jax.nn.relu(x)
    x = self.ff_fc2(x)
    x = x + x_res
    return x


class LConvStack(hk.Module):
  def __init__(self, num_layers, dim, kernel_size, num_heads, dropout_rate=0.1, is_training=True):
    super().__init__()
    self.layers = [
        LConvBlock(dim, kernel_size, num_heads, dropout_rate, is_training)
        for _ in range(num_layers)
    ]
    self.layer_norm = hk.LayerNorm(-1, True, True)

  def __call__(self, x):
    for f in self.layers:
      x = f(x)
    x = self.layer_norm(x)
    return x


class TokenEncoder(hk.Module):
  """Encode phonemes/text to vector"""

  def __init__(self, vocab_size, dim, dropout_rate, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.dropout_rate = dropout_rate
    self.embed = hk.Embed(vocab_size, dim)
    self.conv1 = hk.Conv1D(dim, 5, padding='SAME')
    self.conv2 = hk.Conv1D(dim, 5, padding='SAME')
    self.conv3 = hk.Conv1D(dim, 5, padding='SAME')
    self.bn1 = hk.BatchNorm(True, True, 0.99)
    self.bn2 = hk.BatchNorm(True, True, 0.99)
    self.bn3 = hk.BatchNorm(True, True, 0.99)
    self.lconv_stack = LConvStack(6, dim, 17, 8, 0.1, is_training)

  def __call__(self, x, lengths):
    x = self.embed(x)
    x = jax.nn.relu(self.bn1(self.conv1(x), is_training=self.is_training))
    x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if self.is_training else x
    x = jax.nn.relu(self.bn2(self.conv2(x), is_training=self.is_training))
    x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if self.is_training else x
    x = jax.nn.relu(self.bn3(self.conv3(x), is_training=self.is_training))
    x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if self.is_training else x
    x = posenc(x)
    x = self.lconv_stack(x)
    return x


class DurationModel(hk.Module):
  """Duration model of phonemes."""

  def __init__(self, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.encoder = TokenEncoder(FLAGS.vocab_size, FLAGS.duration_lstm_dim,
                                FLAGS.duration_embed_dropout_rate, is_training)
    self.projection = hk.Sequential([
        hk.Linear(FLAGS.duration_lstm_dim),
        jax.nn.gelu,
        hk.Linear(1),
    ])

  def __call__(self, inputs: DurationInput):
    x = self.encoder(inputs.phonemes, inputs.lengths)
    x = jnp.squeeze(self.projection(x), axis=-1)
    x = jax.nn.softplus(x)
    return x


class AcousticModel(hk.Module):
  """Predict melspectrogram from aligned phonemes"""

  def __init__(self, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.encoder = TokenEncoder(FLAGS.vocab_size, FLAGS.acoustic_encoder_dim, 0.5, is_training)
    self.residual_projection = hk.Linear(FLAGS.acoustic_decoder_dim)
    self.residual_stack = LConvStack(5, FLAGS.acoustic_decoder_dim, 17, 8, 0.1, is_training=is_training)

    self.decoder_stack = [
        LConvBlock(FLAGS.acoustic_decoder_dim, 17, 8, 0.1, is_training=is_training)
        for _ in range(6)
    ]
    self.upsample_projection = hk.Linear(FLAGS.acoustic_decoder_dim)
    self.decoder_projection = [hk.Linear(FLAGS.mel_dim) for _ in range(6)]
    self.vae_projection = hk.Linear(FLAGS.vae_dim * 2)

  def upsample(self, x, durations, L):
    ruler = jnp.arange(0, L)[None, :]  # B, L
    end_pos = jnp.cumsum(durations, axis=1)
    mid_pos = end_pos - durations/2  # B, T

    d2 = jnp.square((mid_pos[:, None, :] - ruler[:, :, None])) / 10.
    w = jax.nn.softmax(-d2, axis=-1)
    hk.set_state('attn', w)
    x = jnp.einsum('BLT,BTD->BLD', w, x)
    return x

  def residual_encoder(self, durations, mels):
    mels = posenc(mels)
    mels = self.residual_projection(mels)
    mels = self.residual_stack(mels)
    B, L, D = mels.shape
    ruler = jnp.arange(0, L)[None, :, None]
    end_frame = jnp.cumsum(durations, axis=1)
    start_frame = end_frame - durations
    end_frame = jnp.ceil(end_frame)
    start_frame = jnp.floor(start_frame)
    mask1 = ruler >= start_frame[:, None, :]
    mask2 = ruler <= end_frame[:, None, :]
    attn = jnp.logical_and(mask1, mask2)
    mels = jnp.einsum('BLT,BLD->BTD', attn, mels)
    mels = mels / (jnp.sum(attn, axis=1)[:, :, None] + 1e-3)
    return mels

  def vae(self, mels):
    params = self.vae_projection(mels)
    mean, logstd = jnp.split(params, 2, axis=-1)
    noise = jax.random.normal(hk.next_rng_key(), shape=mean.shape)
    v = noise * jnp.exp(logstd) + mean
    return v, (mean, logstd)

  def __call__(self, inputs: AcousticInput):
    x = self.encoder(inputs.phonemes, inputs.lengths)

    res = self.residual_encoder(inputs.durations, inputs.mels)
    res, (vae_mean, vae_logstd) = self.vae(res)

    x = jnp.concatenate((x, res), axis=-1)
    x = self.upsample(x, inputs.durations, inputs.mels.shape[1])
    x = self.upsample_projection(x)
    x = posenc(x)

    out = []
    for f, p in zip(self.decoder_stack, self.decoder_projection):
      x = f(x)
      out.append(p(x))

    return out, (vae_mean, vae_logstd)
