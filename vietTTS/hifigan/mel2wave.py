import json
import os
import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.io.wavfile import write

from .model import Generator


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


MAX_WAV_VALUE = 32768.0

config_file = 'assets/hifigan/config.json'
with open(config_file) as f:
  data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)

print(h)


@hk.transform_with_state
def forward(x):
  net = Generator(h)
  return net(x)


rng = next(hk.PRNGSequence(42))

with open('hk_hifi.pickle', 'rb') as f:
  pp = pickle.load(f)
aux = {}
x = np.fromfile('clip.mel', dtype=np.float32).reshape((1, -1, 80))
print(x.shape)
wav, aux = forward.apply(pp, aux, rng, x)
wav = jnp.squeeze(wav)
audio = jax.device_get(wav)
audio = audio * MAX_WAV_VALUE
audio = audio.astype('int16')
output_file = 'clip.wav'
write(output_file, h.sampling_rate, audio)
print(output_file)
