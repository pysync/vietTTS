from argparse import Namespace
from pathlib import Path
from random import sample
from typing import NamedTuple

from jax.numpy import ndarray


class FLAGS(Namespace):
  duration_lstm_dim = 256
  vocab_size = 256
  duration_embed_dropout_rate = 0.5
  num_training_steps = 100_000
  postnet_dim = 512
  acoustic_decoder_dim = 512
  acoustic_encoder_dim = 256

  # dataset
  max_phoneme_seq_len = 128
  max_wave_len = 1024 * 64 * 4


  # dsp
  mel_dim = 160
  n_fft = 1024 
  sample_rate = 16000

  # training
  batch_size = 32
  learning_rate = 1e-4
  max_grad_norm = 1.0

  # ckpt
  ckpt_dir = Path('assets/reinfo/nat')
  # data_dir = Path('assets/reinfo/nat/content/aligned_reinfo')
  data_dir = Path('test_data')


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