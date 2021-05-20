from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from einops.einops import rearrange
from jax.numpy import ndarray

from .config import FLAGS, AcousticInput, DurationInput


class TokenEncoder(hk.Module):
  """Encode phonemes/text to vector"""

  def __init__(self, vocab_size, lstm_dim, dropout_rate, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.embed = hk.Embed(vocab_size, lstm_dim)
    self.conv1 = hk.Conv1D(lstm_dim, 3, padding='SAME', with_bias=False)
    self.conv2 = hk.Conv1D(lstm_dim, 3, padding='SAME', with_bias=False)
    self.conv3 = hk.Conv1D(lstm_dim, 3, padding='SAME', with_bias=False)
    self.bn1 = hk.BatchNorm(True, True, 0.9)
    self.bn2 = hk.BatchNorm(True, True, 0.9)
    self.bn3 = hk.BatchNorm(True, True, 0.9)
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
    self.projection = hk.Linear(256)
    self.lstm = hk.LSTM(FLAGS.duration_lstm_dim)

  def inference(self, inputs: DurationInput):
    x = self.encoder(inputs.phonemes, inputs.lengths)
    B = 1

    def loop_fn(x, state):
      hx, prev_out = state
      x = jnp.concatenate((x, prev_out), axis=-1)
      x, hx = self.lstm(x, hx)
      x = self.projection(x)
      llh = x
      idx = jnp.argmax(x, axis=-1)[..., None]
      x = idx.astype(jnp.float32) / 100.
      return (x, llh), (hx, x)
    (x, llh), _ = hk.dynamic_unroll(loop_fn, x, (self.lstm.initial_state(B), jnp.zeros((1, 1))), time_major=False)
    llh = jax.nn.softmax(llh[0], axis=-1)
    return x[..., 0], llh

  def __call__(self, inputs: DurationInput):
    x = self.encoder(inputs.phonemes, inputs.lengths)
    B, L = inputs.durations.shape
    d = jnp.concatenate((jnp.zeros((B, 1)),  inputs.durations[:, :-1]), axis=1)
    x = jnp.concatenate((x, d[..., None]), axis=-1)
    x, hx = hk.dynamic_unroll(self.lstm, x, self.lstm.initial_state(B), time_major=False)
    x = self.projection(x)
    x = jax.nn.log_softmax(x, axis=-1)
    return x


class AcousticModel(hk.Module):
  """Predict melspectrogram from aligned phonemes"""

  def __init__(self, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.encoder = TokenEncoder(FLAGS.vocab_size, FLAGS.acoustic_encoder_dim, 0.5, is_training)
    self.decoder = hk.deep_rnn_with_skip_connections([
        hk.LSTM(FLAGS.acoustic_decoder_dim),
        hk.LSTM(FLAGS.acoustic_decoder_dim)
    ])
    self.projection = hk.Linear(FLAGS.mel_dim)

    # prenet
    self.prenet_fc1 = hk.Linear(256, with_bias=False)
    self.prenet_bn1 = hk.BatchNorm(True, True, 0.9)
    self.prenet_fc2 = hk.Linear(256, with_bias=False)
    self.prenet_bn2 = hk.BatchNorm(True, True, 0.9)
    # posnet
    self.postnet_convs = [hk.Conv1D(FLAGS.postnet_dim, 5, with_bias=False)
                          for _ in range(4)] + [hk.Conv1D(FLAGS.mel_dim, 5)]
    self.postnet_bns = [hk.BatchNorm(True, True, 0.9) for _ in range(4)] + [None]

  def prenet(self, x, dropout=0.5):
    # use batchnorm as in mozilla TTS
    x = jax.nn.relu(self.prenet_bn1(self.prenet_fc1(x), is_training=self.is_training))
    x = jax.nn.relu(self.prenet_bn2(self.prenet_fc2(x), is_training=self.is_training))
    return x

  def upsample(self, x, durations, L):
    ruler = jnp.arange(0, L)[None, :]  # B, L
    end_pos = jnp.cumsum(durations, axis=1)
    mid_pos = end_pos - durations/2  # B, T

    d2 = jnp.square((mid_pos[:, None, :] - ruler[:, :, None])) / 10.
    w = jax.nn.softmax(-d2, axis=-1)
    hk.set_state('attn', w)
    x = jnp.einsum('BLT,BTD->BLD', w, x)
    return x

  def postnet(self, mel: ndarray) -> ndarray:
    x = mel
    for conv, bn in zip(self.postnet_convs, self.postnet_bns):
      x = conv(x)
      if bn is not None:
        x = bn(x, is_training=self.is_training)
        x = jnp.tanh(x)
      x = hk.dropout(hk.next_rng_key(), 0.5, x) if self.is_training else x
    return x

  def inference(self, tokens, durations, n_frames):
    B, L = tokens.shape
    lengths = jnp.array([L], dtype=jnp.int32)
    x = self.encoder(tokens, lengths)
    x = self.upsample(x, durations, n_frames)

    def loop_fn(inputs, state):
      cond = inputs
      prev_mel, hxcx = state
      prev_mel = self.prenet(prev_mel[:, None])[:, 0]
      x = jnp.concatenate((cond, prev_mel), axis=-1)
      x, new_hxcx = self.decoder(x, hxcx)
      x = jnp.concatenate((cond, x), axis=-1)
      x = self.projection(x)
      return x, (x, new_hxcx)

    state = (
        jnp.zeros((B, FLAGS.mel_dim), dtype=jnp.float32),
        self.decoder.initial_state(B)
    )
    x, _ = hk.dynamic_unroll(loop_fn, x, state, time_major=False)
    residual = self.postnet(x)
    return x + residual

  def __call__(self, inputs: AcousticInput):
    x = self.encoder(inputs.phonemes, inputs.lengths)
    x = self.upsample(x, inputs.durations, inputs.mels.shape[1])
    up_x = x
    mels = self.prenet(inputs.mels)
    x = jnp.concatenate((x, mels), axis=-1)
    B, L, D = x.shape
    hx = self.decoder.initial_state(B)

    def zoneout_decoder(inputs, prev_state):
      x, mask = inputs
      x, state = self.decoder(x, prev_state)
      state = jax.tree_multimap(lambda m, s1, s2: s1*m + s2*(1-m), mask, prev_state, state)
      return x, state

    mask = jax.tree_map(lambda x: jax.random.bernoulli(hk.next_rng_key(), 0.1, (B, L, x.shape[-1])), hx)
    x, _ = hk.dynamic_unroll(zoneout_decoder, (x, mask), hx, time_major=False)
    x = jnp.concatenate((up_x, x), axis=-1)
    x = self.projection(x)
    residual = self.postnet(x)
    return x, x + residual
