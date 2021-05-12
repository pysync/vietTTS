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

    self.glu_fc = hk.Linear(dim*2)
    self.lconv = LightConv(dim, kernel_size, num_heads, dropout_rate, is_training)
    self.layernorm1 = hk.LayerNorm(1, True, True)

    self.ff_fc1 = hk.Linear(dim*4)
    self.ff_fc2 = hk.Linear(dim)

    self.layernorm2 = hk.LayerNorm(1, True, True)

  def __call__(self, x):
    x_res = x
    x = self.glu_fc(x)
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = x1 * jax.nn.sigmoid(x2)
    x = self.lconv(x)
    x = self.layernorm1(x + x_res)

    x_res = x

    x = self.ff_fc1(x)
    x = jax.nn.relu(x)
    x = self.ff_fc2(x)

    x = self.layernorm2(x + x_res)
    return x


class TokenEncoder(hk.Module):
  """Encode phonemes/text to vector"""

  def __init__(self, vocab_size, lstm_dim, dropout_rate, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.embed = hk.Embed(vocab_size, lstm_dim)
    self.conv1 = hk.Conv1D(lstm_dim, 3, padding='SAME')
    self.conv2 = hk.Conv1D(lstm_dim, 3, padding='SAME')
    self.conv3 = hk.Conv1D(lstm_dim, 3, padding='SAME')
    self.bn1 = hk.BatchNorm(True, True, 0.99)
    self.bn2 = hk.BatchNorm(True, True, 0.99)
    self.bn3 = hk.BatchNorm(True, True, 0.99)
    self.lstm_fwd = hk.LSTM(lstm_dim)
    self.lstm_bwd = hk.ResetCore(hk.LSTM(lstm_dim))
    self.dropout_rate = dropout_rate

  def __call__(self, x, lengths):
    x = self.embed(x)
    x = jax.nn.relu(self.bn1(self.conv1(x), is_training=self.is_training))
    x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if self.is_training else x
    x = jax.nn.relu(self.bn2(self.conv2(x), is_training=self.is_training))
    x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if self.is_training else x
    x = jax.nn.relu(self.bn3(self.conv3(x), is_training=self.is_training))
    x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if self.is_training else x
    B, L, D = x.shape
    mask = jnp.arange(0, L)[None, :] >= (lengths[:, None] - 1)
    h0c0_fwd = self.lstm_fwd.initial_state(B)
    new_hx_fwd, new_hxcx_fwd = hk.dynamic_unroll(self.lstm_fwd, x, h0c0_fwd, time_major=False)
    x_bwd, mask_bwd = jax.tree_map(lambda x: jnp.flip(x, axis=1), (x, mask))
    h0c0_bwd = self.lstm_bwd.initial_state(B)
    new_hx_bwd, new_hxcx_bwd = hk.dynamic_unroll(self.lstm_bwd, (x_bwd, mask_bwd), h0c0_bwd, time_major=False)
    x = jnp.concatenate((new_hx_fwd, jnp.flip(new_hx_bwd, axis=1)), axis=-1)
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
    self.residual_stack = [
        LConvBlock(FLAGS.acoustic_decoder_dim, 17, 8, 0.1, is_training=is_training)
        for _ in range(5)
    ]
    self.residual_stack.insert(0, hk.Linear(FLAGS.acoustic_decoder_dim))

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
    for f in self.residual_stack:
      mels = f(mels)
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
    mels = mels / (jnp.sum(attn, axis=1)[:, :, None])
    return mels

  def vae(self, mels):
    params = self.vae_projection(mels)
    mean, logstd = jnp.split(params, 2, axis=-1)
    noise = jax.random.normal(hk.next_rng_key(), shape=mean.shape)
    v = noise = noise * jnp.exp(logstd) + mean
    return v, (mean, logstd)

  def __call__(self, inputs: AcousticInput):
    x = self.encoder(inputs.phonemes, inputs.lengths)

    res = self.residual_encoder(inputs.durations, inputs.mels)
    res, (vae_mean, vae_logstd) = self.vae(res)

    x = jnp.concatenate((x, res), axis=-1)
    x = self.upsample(x, inputs.durations, inputs.mels.shape[1])
    x = self.upsample_projection(x)

    out = []
    for f, p in zip(self.decoder_stack, self.decoder_projection):
      x = f(x)
      out.append(p(x))

    return out, (vae_mean, vae_logstd)
