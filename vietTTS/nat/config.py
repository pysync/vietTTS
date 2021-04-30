from argparse import Namespace
from pathlib import Path
from typing import NamedTuple

from jax.numpy import ndarray


class FLAGS(Namespace):
  duration_lstm_dim = 256
  vocab_size = 256
  duration_embed_dropout_rate = 0.5
  num_training_steps = 100_000
  max_phoneme_seq_len = 128

  # training
  batch_size = 32
  learning_rate = 1e-4
  max_grad_norm = 1.0

  # ckpt
  ckpt_dir = Path('assets/reinfo/nat')
  data_dir = Path('assets/reinfo/nat/content/aligned_reinfo')


class DurationInput(NamedTuple):
  phonemes: ndarray
  lengths: ndarray
  durations: ndarray
