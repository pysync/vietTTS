from argparse import Namespace
from pathlib import Path
from typing import NamedTuple

from jax.numpy import ndarray


class FLAGS(Namespace):
  duration_lstm_dim = 1024
  vocab_size = 512
  duration_embed_dropout_rate = 0.5
  num_training_steps = 1_00_000

  learning_rate = 1e-4
  # ckpt
  ckpt_dir = Path('assets/reinfo/nat')


class DurationInput(NamedTuple):
  phonemes: ndarray
  lengths: ndarray
