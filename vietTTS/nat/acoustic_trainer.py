from typing import Deque

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from vietTTS.nat.config import AcousticInput
from vietTTS.tacotron.dsp import MelFilter

from .config import FLAGS, AcousticInput
from .model import AcousticModel
from .utils import print_flags
from .data_loader import load_textgrid_wav



@hk.transform_with_state
def net(x): return AcousticModel(is_training=True)(x)


def loss_fn(params, aux, rng, inputs: AcousticInput):
  melfilter = MelFilter(FLAGS.sample_rate, FLAGS.n_fft, FLAGS.mel_dim)
  mels=melfilter(inputs.wavs.astype(jnp.float32) / (2**15))
  B, L, D= mels.shape
  inp_mels = jnp.concatenate( 
    (
      jnp.zeros((B, 1, D), dtype=jnp.float32), 
      mels[:, :-1, :]
    ), axis=1)


  n_frames = inputs.durations / 10 * FLAGS.sample_rate / (FLAGS.n_fft//4)
  inputs = inputs._replace(mels = inp_mels, durations=n_frames)
  (mel1_hat, mel2_hat), new_aux = net.apply(params, aux, rng, inputs)
  loss = jnp.mean(jnp.square(mel1_hat - mels) + jnp.square(mel2_hat - mels))
  return loss, new_aux


loss_vag = jax.value_and_grad(loss_fn, has_aux=True)


optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(FLAGS.learning_rate)
)


@jax.jit
def update(params, aux, rng, optim_state, inputs):
  rng, new_rng = jax.random.split(rng)
  (loss, new_aux), grads = loss_vag(params, aux, rng, inputs)
  updates, new_optim_state = optimizer.update(grads, optim_state, params)
  new_params = optax.apply_updates(updates, params)
  return loss, (new_params, new_aux, new_rng, new_optim_state)


def initial_state(batch):
  rng = jax.random.PRNGKey(42)
  params, aux = hk.transform_with_state(lambda x: AcousticModel(True)(x)).init(rng, batch)
  optim_state = optimizer.init(params)
  return params, aux, rng, optim_state


def train():
  train_data_iter = load_textgrid_wav(FLAGS.data_dir, FLAGS.max_phoneme_seq_len, 2, FLAGS.max_wave_len, 'train')
  batch = next(train_data_iter)
  melfilter = MelFilter(FLAGS.sample_rate, FLAGS.n_fft, FLAGS.mel_dim)
  batch = batch._replace(mels=melfilter(batch.wavs.astype(jnp.float32) / (2**15)))
  params, aux, rng, optim_state = initial_state(batch)
  losses = Deque(maxlen=1000)
  val_losses = Deque(maxlen=1000)

  last_step = -1
  for step in range(last_step + 1, FLAGS.num_training_steps + 1):
    loss, (params, aux, rng, optim_state) = update(params, aux, rng, optim_state, batch)
    losses.append(loss)

    if step % 100 == 0:
      loss = sum(losses).item() / len(losses)
      print(f'step {step}  train loss {loss:.3f}')


if __name__ == '__main__':
  print_flags(FLAGS.__dict__)
  if not FLAGS.ckpt_dir.exists():
    print('Create checkpoint dir at', FLAGS.ckpt_dir)
    FLAGS.ckpt_dir.mkdir(parents=True, exist_ok=True)
  train()
