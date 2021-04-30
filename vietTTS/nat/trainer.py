from functools import partial
from typing import Deque

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from vietTTS.nat.config import DurationInput

from .config import *
from .data_loader import text_mel_data_loader
from .model import DurationModel
from .utils import *


def loss_fn(params, aux, rng, x: DurationInput, is_training=True):
  @hk.transform_with_state
  def net(x):
    return DurationModel(is_training=is_training)(x)
  durations, aux = net.apply(params, aux, rng, x)
  mask = jnp.arange(0, x.phonemes.shape[1])[None, :] < x.lengths[:, None]
  masked_loss = jnp.square(durations - x.durations) * mask
  loss = jnp.sum(masked_loss) / jnp.sum(mask)
  return loss


val_loss_fn = jax.jit(partial(loss_fn, is_training=False))

loss_vag = jax.value_and_grad(loss_fn, has_aux=True)

scheduler = optax.exponential_decay(1., 1000, 0.99, staircase=True, end_value=1e-6)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.),
    optax.adam(FLAGS.learning_rate),
    optax.scale_by_schedule(scheduler)
)


def update(params, aux, rng, optim_state, inputs: DurationInput):
  rng, new_rng = jax.random.split(rng)
  (loss, new_aux), grads = loss_vag(params, aux, rng, inputs)
  updates, new_optim_state = optimizer.update(grads, optim_state, params)
  new_params = optax.apply_updates(params, updates)
  return loss, (new_params, new_aux, new_rng, new_optim_state)


def train():
  train_data_iter = text_mel_data_loader(FLAGS.data_dir, mode='train')
  val_data_iter = text_mel_data_loader(FLAGS.data_dir, mode='val')
  last_step = -1
  latest_ckpt = load_latest_ckpt(FLAGS.ckpt_dir)
  losses = Deque(maxlen=1000)
  val_losses = Deque(maxlen=1000)
  if latest_ckpt is not None:
    last_step, param, aux, rng, optim_state = latest_ckpt
  for step in range(last_step+1, 1+FLAGS.num_training_steps):
    batch = next(train_data_iter)
    loss, (params, aux, rng, optim_state) = update(params, aux, rng, optim_state, batch)
    losses.append(loss)

    if step % 10 == 0:
      val_loss, aux = val_loss_fn(params, aux, rng, next(val_data_iter))
      val_losses.append(val_loss)

    if step % 1000 == 0:
      loss = sum(losses).item() / len(losses)
      val_loss = sum(val_losses).item() / len(val_losses)
      print(f'  {step:07d}  train loss {loss:.3f}   val_loss {val_loss:.3f}')
      save_ckpt(step, params, aux, rng, optim_state, ckpt_dir=FLAGS.ckpt_dir)


if __name__ == '__main__':
  if not FLAGS.ckpt_dir.exists():
    print('Create checkpoint dir at', FLAGS.ckpt_dir)
    FLAGS.ckpt_dir.mkdir(parents=True, exist_ok=True)
  train()
