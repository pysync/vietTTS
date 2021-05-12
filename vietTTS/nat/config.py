from argparse import Namespace
from pathlib import Path
from random import sample
from typing import NamedTuple

from jax.numpy import ndarray


class TrainingSchedule(NamedTuple):
  end_step: int
  learning_rate: float
  beta: float


class FLAGS(Namespace):
  duration_lstm_dim = 256
  vocab_size = 256
  duration_embed_dropout_rate = 0.5
  num_training_steps = 100_000
  acoustic_decoder_dim = 256
  acoustic_encoder_dim = 256
  vae_dim = 8

  # dataset
  max_phoneme_seq_len = 128*3
  max_wave_len = 1024 * 64 * 2

  # dsp
  mel_dim = 80
  n_fft = 1024
  sample_rate = 16000
  fmin = 0.0
  fmax = 8000

  # training
  batch_size = 64
  learning_rate = 1e-3
  max_grad_norm = 1.0

  _acoustic_schedule = [
      TrainingSchedule(1000,   1e-7, 0.0),
      TrainingSchedule(2000,   1e-6, 0.0),
      TrainingSchedule(5000,   1e-5, 0.0),
      TrainingSchedule(10000,  5e-5, 1e-4),
      TrainingSchedule(20000,  3e-4, 1e-3),
      TrainingSchedule(40000,  1e-3, 1e-2),
      TrainingSchedule(50000,  1e-3, 1e-1),
      TrainingSchedule(100000, 1e-3, 1e-0),
      TrainingSchedule(200000, 1e-4, 1e-0),
      TrainingSchedule(400000, 2e-5, 1e-0),
      TrainingSchedule(500000, 1e-5, 1e-0),
  ]

  # ckpt
  ckpt_dir = Path('assets/infore/nat')
  data_dir = Path('assets/infore/data')
  data_dir = Path('train_data')


class DurationInput(NamedTuple):
  phonemes: ndarray
  lengths: ndarray
  durations: ndarray


class AcousticInput(NamedTuple):
  phonemes: ndarray
  lengths: ndarray
  durations: ndarray
  wavs: ndarray
  wav_lengths: ndarray
  mels: ndarray
