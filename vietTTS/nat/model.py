from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from einops.einops import rearrange
from haiku._src.data_structures import K
from jax.numpy import ndarray
from vietTTS.tacotron.config import FLAGS

from .config import FLAGS, AcousticInput, DurationInput


class LSTMState(NamedTuple):
  hidden: jnp.ndarray
  cell: jnp.ndarray
  rng: jnp.ndarray


def add_batch(nest, batch_size: Optional[int]):
  def broadcast(x): return jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree_map(broadcast, nest)


class LSTM(hk.RNNCore):

  def __init__(self, hidden_size: int, is_training=True, name: Optional[str] = None):
    super().__init__(name=name)
    self.hidden_size = hidden_size
    self.is_training = is_training

  def __call__(self, inputs: jnp.ndarray, prev_state: LSTMState,) -> Tuple[jnp.ndarray, LSTMState]:
    if len(inputs.shape) > 2 or not inputs.shape:
      raise ValueError("LSTM input must be rank-1 or rank-2.")
    x_and_h = jnp.concatenate([inputs, prev_state.hidden], axis=-1)
    gated = hk.Linear(4 * self.hidden_size)(x_and_h)
    i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)
    f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.
    c = f * prev_state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
    h = jax.nn.sigmoid(o) * jnp.tanh(c)

    if self.is_training:
      rng1, rng_next = jax.random.split(prev_state.rng[0], 2)
      mask = jax.random.bernoulli(rng1, 0.1, (2,) + h.shape)
      h = mask[0] * prev_state.hidden + (1 - mask[0]) * h
      c = mask[1] * prev_state.cell + (1 - mask[1]) * c
      rng_next = add_batch(rng_next, h.shape[0])
    else:
      rng_next = prev_state.rng
    return h, LSTMState(h, c, rng_next)

  def initial_state(self, batch_size: Optional[int]) -> LSTMState:
    state = LSTMState(hidden=jnp.zeros([self.hidden_size]),
                      cell=jnp.zeros([self.hidden_size]),
                      rng=hk.next_rng_key())
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state


class TokenEncoder(hk.Module):
  """Encode phonemes/text to vector"""

  def __init__(self, vocab_size, lstm_dim, dropout_rate, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.embed = hk.Embed(vocab_size, lstm_dim)
    self.lstm_fwd = LSTM(lstm_dim, is_training=is_training)
    self.lstm_bwd = hk.ResetCore(LSTM(lstm_dim, is_training=is_training))
    self.dropout_rate = dropout_rate

  def __call__(self, x, lengths):
    x = self.embed(x)
    x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if self.is_training else x
    B, L, D = x.shape
    mask = jnp.arange(0, L)[None, :] < (lengths[:, None] - 1)
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
    self.projection = hk.Linear(1)

  def __call__(self, inputs: DurationInput):
    x = self.encoder(inputs.phonemes, inputs.lengths)
    x = jnp.squeeze(self.projection(x), axis=-1)
    x = jax.nn.relu(x)
    return x


class AcousticModel(hk.Module):
  """Predict melspectrogram from aligned phonemes"""

  def __init__(self, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.encoder = TokenEncoder(FLAGS.vocab_size, FLAGS.acoustic_encoder_dim, 0.5, is_training)
    self.decoder = hk.deep_rnn_with_skip_connections([
        LSTM(FLAGS.acoustic_decoder_dim, is_training=is_training),
        LSTM(FLAGS.acoustic_decoder_dim, is_training=is_training)
    ])
    self.projection = hk.Linear(FLAGS.mel_dim)

    # prenet
    # self.prenet_fc1 = hk.Linear(256, with_bias=True)
    # self.prenet_fc2 = hk.Linear(256, with_bias=True)
    # posnet
    self.postnet_convs = [hk.Conv1D(FLAGS.postnet_dim, 5) for _ in range(4)] + [hk.Conv1D(FLAGS.mel_dim, 5)]
    self.postnet_bns = [hk.BatchNorm(True, True, 0.99) for _ in range(4)] + [None]

  def prenet(self, x, dropout=0.5):
    # x = jax.nn.relu(self.prenet_fc1(x))
    # x = hk.dropout(hk.next_rng_key(), dropout, x) if dropout > 0 else x
    # x = jax.nn.relu(self.prenet_fc2(x))
    # x = hk.dropout(hk.next_rng_key(), dropout, x) if dropout > 0 else x
    return x

  def upsample(self, x, durations, L):
    ruler = jnp.arange(0, L)[None, :]  # B, L
    end_pos = jnp.cumsum(durations, axis=1)
    mid_pos = end_pos - durations/2  # B, T

    d2 = jnp.square((mid_pos[:, None, :] - ruler[:, :, None])) / 10.
    w = jax.nn.softmax(-d2, axis=-1)
    # import matplotlib.pyplot as plt
    # plt.imshow(w[0].T)
    # plt.savefig('att.png')
    # plt.close()
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
      prev_mel = self.prenet(prev_mel)
      x = jnp.concatenate((cond, prev_mel), axis=-1)
      x, new_hxcx = self.decoder(x, hxcx)
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
    mels = self.prenet(inputs.mels)
    x = jnp.concatenate((x, mels), axis=-1)
    B, L, D = x.shape
    hx = self.decoder.initial_state(B)
    x, _ = hk.dynamic_unroll(self.decoder, x, hx, time_major=False)
    x = self.projection(x)
    residual = self.postnet(x)
    return x, x + residual
