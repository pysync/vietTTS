import json
import os
import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import torch
from scipy.io.wavfile import write

from .model import Generator


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


MAX_WAV_VALUE = 32768.0


MAX_WAV_VALUE = 32768.0


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


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


mel = jnp.zeros((1, 80, 20))
rng = next(hk.PRNGSequence(42))
params, aux = forward.init(rng, mel)
print(jax.tree_map(lambda x: x.shape, params))

with open('hk_hifi.pickle', 'rb') as f:
  pp = pickle.load(f)
torch.manual_seed(42)
x = torch.zeros((1, 80, 60)).contiguous().numpy()
x = torch.load('m.mel').numpy()
# o = jnp.swapaxes(x, 1, 2)
o = x
o, aux = forward.apply(pp, aux, rng, o)
o = jnp.squeeze(o)
print(o.shape)
print(o[:100])
# import pdb; pdb.set_trace()
print(x[0, 0, :10])
print(jnp.sum(o))

with open('hk.p', 'wb') as f:
  pickle.dump(jax.device_get(o), f)

audio = jax.device_get(o)
audio = audio * MAX_WAV_VALUE
audio = audio.astype('int16')
output_file = 'rand_generated_e2e.wav'
write(output_file, h.sampling_rate, audio)
print(output_file)
