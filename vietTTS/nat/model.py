import haiku as hk
import jax
import jax.numpy as jnp
from vietTTS.tacotron.config import FLAGS

from .config import FLAGS, DurationInput


class DurationModel(hk.Module):
  """Duration model of phonemes."""

  def __init__(self, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.embed = hk.Embed(FLAGS.vocab_size, FLAGS.duration_lstm_dim)
    self.lstm_fwd = hk.LSTM(FLAGS.duration_lstm_dim)
    self.lstm_bwd = hk.ResetCore(hk.LSTM(FLAGS.duration_lstm_dim))
    self.projection = hk.Linear(1)

  def __call__(self, inputs: DurationInput):
    x = self.embed(inputs.phonemes)
    x = hk.dropout(hk.next_rng_key(), FLAGS.duration_embed_dropout_rate, x) if self.is_training else x
    B, L, D = x.shape
    mask = jnp.arange(0, L)[None, :] < (inputs.lengths[:, None] - 1)

    h0c0_fwd = self.lstm_fwd.initial_state(B)
    new_hx_fwd, new_hxcx_fwd = hk.dynamic_unroll(self.lstm_fwd, x, h0c0_fwd, time_major=False)
    x_bwd, mask_bwd = jax.tree_map(lambda x: jnp.flip(x, axis=1), (x, mask))
    h0c0_bwd = self.lstm_bwd.initial_state(B)
    new_hx_bwd, new_hxcx_bwd = hk.dynamic_unroll(self.lstm_bwd, (x_bwd, mask_bwd), h0c0_bwd, time_major=False)
    x = jnp.concatenate(
        (new_hx_fwd, jnp.flip(new_hx_bwd, axis=1)),
        axis=-1
    )
    x = self.projection(x)
    x = jax.nn.relu(x)
    return x
