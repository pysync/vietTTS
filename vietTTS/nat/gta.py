import os
import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
from vietTTS.nat.config import AcousticInput
from vietTTS.tacotron.dsp import MelFilter

from .config import FLAGS, AcousticInput
from .data_loader import load_textgrid_wav
from .model import AcousticModel


@hk.transform_with_state
def net(x): return AcousticModel(is_training=True)(x)


@hk.transform_with_state
def val_net(x): return AcousticModel(is_training=False)(x)


def loss_fn(params, aux, rng, inputs: AcousticInput, is_training=True):
  melfilter = MelFilter(FLAGS.sample_rate, FLAGS.n_fft, FLAGS.mel_dim, FLAGS.fmin, FLAGS.fmax)
  mels = melfilter(inputs.wavs.astype(jnp.float32) / (2**15))
  B, L, D = mels.shape
  inp_mels = jnp.concatenate((jnp.zeros((B, 1, D), dtype=jnp.float32), mels[:, :-1, :]), axis=1)
  n_frames = inputs.durations / 10 * FLAGS.sample_rate / (FLAGS.n_fft//4)
  inputs = inputs._replace(mels=inp_mels, durations=n_frames)
  (mel1_hat, mel2_hat), new_aux = (net if is_training else val_net).apply(params, aux, rng, inputs)
  loss1 = (jnp.square(mel1_hat - mels) + jnp.square(mel2_hat - mels)) / 2
  loss2 = (jnp.abs(mel1_hat - mels) + jnp.abs(mel2_hat - mels)) / 2
  loss = jnp.mean((loss1 + loss2)/2, axis=-1)
  mask = (jnp.arange(0, L)[None, :] - 10) < (inputs.wav_lengths // (FLAGS.n_fft // 4))[:, None]
  loss = jnp.sum(loss * mask) / jnp.sum(mask)
  return (loss, new_aux) if is_training else (loss, new_aux, mel2_hat, mels)


train_loss_fn = partial(loss_fn, is_training=True)
val_loss_fn = jax.jit(partial(loss_fn, is_training=False))


def generate_gta(out_dir: Path):
  out_dir.mkdir(parents=True, exist_ok=True)
  data_iter = load_textgrid_wav(FLAGS.data_dir, FLAGS.max_phoneme_seq_len,
                                FLAGS.batch_size, FLAGS.max_wave_len, 'gta')
  ckpt_fn = FLAGS.ckpt_dir / 'acoustic_ckpt_latest.pickle'
  print('Resuming from latest checkpoint at', ckpt_fn)
  with open(ckpt_fn, 'rb') as f:
    dic = pickle.load(f)
    _, params, aux, rng, _ = dic['step'], dic['params'], dic['aux'], dic['rng'], dic['optim_state']

  tr = tqdm(data_iter)
  for names, batch in tr:
    lengths = batch.wav_lengths
    _, _, predicted_mel, _ = val_loss_fn(params, aux, rng, batch)
    mel = jax.device_get(predicted_mel)
    for idx, fn in enumerate(names):
      file = out_dir / f'{fn}.npy'
      tr.write(f'saving to file {file}')
      l = lengths[idx] // (FLAGS.n_fft//4)
      np.save(file, mel[idx, :l].T)
      os.symlink(FLAGS.data_dir.resolve() / f'{fn}.wav', out_dir / f'{fn}.wav', False)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-o', '--output-dir', type=Path, default='gta')
  generate_gta(parser.parse_args().output_dir)
