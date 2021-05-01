import haiku as hk
import jax
import jax.numpy as jnp
from einops.einops import rearrange
from haiku._src.data_structures import K
from jax.numpy import ndarray
from vietTTS.tacotron.config import FLAGS

from .config import FLAGS, DurationInput, AcousticInput


class TokenEncoder(hk.Module):
  """Encode phonemes/text to vector"""

  def __init__(self, vocab_size, lstm_dim, dropout_rate, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.embed = hk.Embed(vocab_size, lstm_dim)
    self.lstm_fwd = hk.LSTM(lstm_dim)
    self.lstm_bwd = hk.ResetCore(hk.LSTM(lstm_dim))
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
        hk.LSTM(FLAGS.acoustic_decoder_dim),
        hk.LSTM(FLAGS.acoustic_decoder_dim)
    ])
    self.projection = hk.Linear(FLAGS.mel_dim)

    self.postnet_convs = [hk.Conv1D(FLAGS.postnet_dim, 5) for _ in range(4)] + [hk.Conv1D(FLAGS.mel_dim, 5)]
    self.postnet_bns = [hk.BatchNorm(True, True, 0.99) for _ in range(4)] + [None]

  def upsample(self, x, durations, L):
    ruler = jnp.arange(0, L)[None, :]  # B, L
    end_pos = jnp.cumsum(durations, axis=1)
    mid_pos = end_pos - durations/2  # B, T

    d2 = jnp.square((mid_pos[:, None, :] - ruler[:, :, None])) / 10.
    w = jax.nn.softmax(-d2, axis=-1)
    # import matplotlib.pyplot as plt 
    # plt.imshow(w[0, 50:150, :50].T)
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

  def __call__(self, inputs:AcousticInput):
    x = self.encoder(inputs.phonemes, inputs.lengths)
    x = self.upsample(x, inputs.durations, inputs.mels.shape[1])
    x = jnp.concatenate((x, inputs.mels), axis=-1)
    B, L, D = x.shape
    hx = self.decoder.initial_state(B)
    x, new_hx = hk.dynamic_unroll(self.decoder, x, hx, time_major=False)
    x = self.projection(x)
    residual = self.postnet(x)
    return x, x + residual
