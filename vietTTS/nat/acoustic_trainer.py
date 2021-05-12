import pickle
from functools import partial
from typing import Deque

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax.interpreters.masking import is_tracing
from tqdm.auto import tqdm
from vietTTS.nat.config import AcousticInput
from vietTTS.tacotron.dsp import MelFilter

from .config import FLAGS, AcousticInput
from .data_loader import load_textgrid_wav
from .model import AcousticModel
from .utils import print_flags


@hk.transform_with_state
def net(x): return AcousticModel(is_training=True)(x)


@hk.transform_with_state
def val_net(x): return AcousticModel(is_training=False)(x)


# @jax.jit
# def val_forward(params, aux, rng, inputs: AcousticInput):
#   melfilter = MelFilter(FLAGS.sample_rate, FLAGS.n_fft, FLAGS.mel_dim, FLAGS.fmin, FLAGS.fmax)
#   mels = melfilter(inputs.wavs.astype(jnp.float32) / (2**15))
#   B, L, D = mels.shape
#   inp_mels = jnp.concatenate((jnp.zeros((B, 1, D), dtype=jnp.float32), mels[:, :-1, :]), axis=1)

#   n_frames = inputs.durations / 10 * FLAGS.sample_rate / (FLAGS.n_fft//4)
#   inputs = inputs._replace(mels=inp_mels, durations=n_frames)
#   (mel1_hat, mel2_hat), new_aux = val_net.apply(params, aux, rng, inputs)
#   return mel1_hat, mel2_hat


def loss_fn(params, aux, rng, inputs: AcousticInput, beta, is_training=True):
  melfilter = MelFilter(FLAGS.sample_rate, FLAGS.n_fft, FLAGS.mel_dim, FLAGS.fmin, FLAGS.fmax)
  mels = melfilter(inputs.wavs.astype(jnp.float32) / (2**15))
  B, L, D = mels.shape
  inp_mels = jnp.concatenate((jnp.zeros((B, 1, D), dtype=jnp.float32), mels[:, :-1, :]), axis=1)
  n_frames = inputs.durations / 10 * FLAGS.sample_rate / (FLAGS.n_fft//4)
  inputs = inputs._replace(mels=inp_mels, durations=n_frames)
  (mel_stack, vae_params), new_aux = (net if is_training else val_net).apply(params, aux, rng, inputs)

  loss = 0.0
  for mel_hat in mel_stack:
    loss = loss + jnp.mean(jnp.abs(mel_hat - mels), axis=-1)
  loss = loss / len(mel_stack)

  # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
  mu, log_std = vae_params
  log_sigma = 2 * log_std
  dkl = 0.5 * jnp.sum(jnp.exp(log_sigma) + jnp.square(mu) - 1. - log_sigma, axis=-1)

  mask = jnp.arange(0, L)[None, :] < (inputs.wav_lengths // (FLAGS.n_fft // 4))[:, None]
  loss = jnp.sum(loss * mask) / jnp.sum(mask)

  mask = jnp.arange(0, dkl.shape[1])[None, :] < inputs.lengths[:, None]
  vae_loss = jnp.sum(dkl * mask) / jnp.sum(mask)

  loss = loss + vae_loss * beta

  return (loss, (vae_loss, new_aux)) if is_training else (loss, vae_loss, new_aux, mel_stack[-1], mels)


train_loss_fn = partial(loss_fn, is_training=True)
val_loss_fn = jax.jit(partial(loss_fn, is_training=False))

loss_vag = jax.value_and_grad(train_loss_fn, has_aux=True)


def make_optimizer(lr):
  return optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adam(lr),
  )


optimizer = make_optimizer(FLAGS.learning_rate)


@partial(jax.jit, static_argnums=[5])
def update(params, aux, rng, optim_state, inputs, schedule):
  rng, new_rng = jax.random.split(rng)
  (loss, (vae_loss, new_aux)), grads = loss_vag(params, aux, rng, inputs, schedule.beta)
  optimizer = make_optimizer(schedule.learning_rate)
  updates, new_optim_state = optimizer.update(grads, optim_state, params)
  new_params = optax.apply_updates(updates, params)
  return (loss, vae_loss), (new_params, new_aux, new_rng, new_optim_state)


def initial_state(batch):
  rng = jax.random.PRNGKey(42)
  n_frames = batch.durations / 10 * FLAGS.sample_rate / (FLAGS.n_fft//4)
  batch = batch._replace(durations=n_frames)
  params, aux = hk.transform_with_state(lambda x: AcousticModel(True)(x)).init(rng, batch)
  optim_state = optimizer.init(params)
  return params, aux, rng, optim_state


def train():
  train_data_iter = load_textgrid_wav(FLAGS.data_dir, FLAGS.max_phoneme_seq_len,
                                      FLAGS.batch_size, FLAGS.max_wave_len, 'train')
  val_data_iter = load_textgrid_wav(FLAGS.data_dir, FLAGS.max_phoneme_seq_len,
                                    FLAGS.batch_size, FLAGS.max_wave_len, 'val')
  melfilter = MelFilter(FLAGS.sample_rate, FLAGS.n_fft, FLAGS.mel_dim, FLAGS.fmin, FLAGS.fmax)
  batch = next(train_data_iter)
  batch = batch._replace(mels=melfilter(batch.wavs.astype(jnp.float32) / (2**15)))
  params, aux, rng, optim_state = initial_state(batch)
  losses = Deque(maxlen=1000)
  val_losses = Deque(maxlen=100)
  vae_losses = Deque(maxlen=1000)

  last_step = -1

  # loading latest checkpoint
  ckpt_fn = FLAGS.ckpt_dir / 'acoustic_ckpt_latest.pickle'
  if ckpt_fn.exists():
    print('Resuming from latest checkpoint at', ckpt_fn)
    with open(ckpt_fn, 'rb') as f:
      dic = pickle.load(f)
      last_step, params, aux, rng, optim_state = dic['step'], dic['params'], dic['aux'], dic['rng'], dic['optim_state']

  tr = tqdm(
      range(last_step + 1, FLAGS.num_training_steps + 1),
      desc='training',
      total=FLAGS.num_training_steps+1,
      initial=last_step+1
  )
  for s in FLAGS._acoustic_schedule:
    if s.end_step > last_step + 1:
      schedule = s
      break

  for step in tr:
    batch = next(train_data_iter)
    (loss, vae_loss), (params, aux, rng, optim_state) = update(params, aux, rng, optim_state, batch, schedule)
    losses.append(loss)
    vae_losses.append(vae_loss)

    if step % 10 == 0:
      val_batch = next(val_data_iter)
      val_loss, val_vae_loss, val_aux, predicted_mel, gt_mel = val_loss_fn(params, aux, rng, val_batch, schedule.beta)
      val_losses.append(val_loss)
      attn = jax.device_get(val_aux['acoustic_model']['attn'][0])
      predicted_mel = jax.device_get(predicted_mel[0])
      gt_mel = jax.device_get(gt_mel[0])

    if step % 1000 == 0:
      loss = sum(losses).item() / len(losses)
      val_loss = sum(val_losses).item() / len(val_losses)
      vae_loss = sum(vae_losses).item() / len(vae_losses)
      tr.write(f'step {step}  train loss {loss:.3f}  val loss {val_loss:.3f}  vae loss {vae_loss:.3f}  lr {schedule.learning_rate:.3e}  beta {schedule.beta:.3e}')

      # saving predicted mels
      plt.figure(figsize=(10, 10))
      plt.subplot(3, 1, 1)
      plt.imshow(predicted_mel.T, origin='lower', aspect='auto',
                 vmin=jnp.min(gt_mel.T).item(), vmax=jnp.max(gt_mel.T).item())
      plt.subplot(3, 1, 2)
      plt.imshow(gt_mel.T, origin='lower', aspect='auto')
      plt.subplot(3, 1, 3)
      plt.imshow(attn.T, origin='lower', aspect='auto')
      plt.tight_layout()
      plt.savefig(FLAGS.ckpt_dir / f'mel_{step}.png')
      plt.close()

      # saving checkpoint
      with open(ckpt_fn, 'wb') as f:
        pickle.dump({'step': step, 'params': params, 'aux': aux, 'rng': rng, 'optim_state': optim_state}, f)

      # update schedule
      for s in FLAGS._acoustic_schedule:
        if s.end_step > step:
          schedule = s
          break


if __name__ == '__main__':
  print_flags(FLAGS.__dict__)
  if not FLAGS.ckpt_dir.exists():
    print('Create checkpoint dir at', FLAGS.ckpt_dir)
    FLAGS.ckpt_dir.mkdir(parents=True, exist_ok=True)
  train()
