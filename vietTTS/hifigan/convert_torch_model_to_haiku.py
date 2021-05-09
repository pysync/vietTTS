from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import glob
import json
import os
import pickle
import shutil

import numpy as np
import torch
from .torch_model import Generator
from scipy.io.wavfile import write


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


MAX_WAV_VALUE = 32768.0

h = None
device = None


def load_checkpoint(filepath, device):
  assert os.path.isfile(filepath)
  print("Loading '{}'".format(filepath))
  checkpoint_dict = torch.load(filepath, map_location=device)
  print("Complete.")
  return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
  pattern = os.path.join(cp_dir, prefix + '*')
  cp_list = glob.glob(pattern)
  if len(cp_list) == 0:
    return ''
  return sorted(cp_list)[-1]


def inference(a):
  generator = Generator(h).to(device)

  state_dict_g = load_checkpoint(a.checkpoint_file, device)
  generator.load_state_dict(state_dict_g['generator'])

  # filelist = os.listdir(a.input_mels_dir)

  os.makedirs(a.output_dir, exist_ok=True)

  generator.eval()
  generator.remove_weight_norm()
  # torch.save(generator.state_dict(), 'no_wn.torch', )
  # print(generator.state_dict().keys())
  hk_map = {}
  for a, b in generator.state_dict().items():
    print(a, b.shape)

    if a.startswith('conv_pre'): a = 'generator/~/conv1_d'
    elif a.startswith('conv_post'): a = 'generator/~/conv1_d_1'
    elif a.startswith('ups.'):
      ii = a.split('.')[1]
      # print()
      a = f'generator/~/ups_{ii}'
    elif a.startswith('resblocks.'):
      _, x, y, z, _ = a.split('.')
      a = f'generator/~/res_block1_{x}/~/{y}_{z}'
    
    # a = a.replace('_0', '')
    print(a, b.shape)
    if a not in hk_map:
      hk_map[a]= {} 
    if len(b.shape) == 1:
      hk_map[a]['b'] = b.numpy()[:, None]
    else:
      if 'ups'  in a:
        x = b.numpy()#hk_map[a]['w'] 
        hk_map[a]['w'] = np.rot90(x, k=1, axes=(0, 2))
        # import pdb; pdb.set_trace()
      elif 'conv' in a:
        hk_map[a]['w'] = np.swapaxes(b.numpy(), 0, 2)
      else:
        hk_map[a]['w'] = b.numpy()

  
  # print(hk_map.keys())
  import pickle
  with open('hk_hifi.pickle', 'wb') as f:
    pickle.dump(hk_map, f)
  with torch.no_grad():
    # for i, filname in enumerate(filelist):
    # x = np.load(os.path.join(a.input_mels_dir, filname))
    # x = np.ran
    torch.manual_seed(42)
    x = torch.zeros((1, 80, 60)).to(device)
    # print(x[0, 0, :10])
    x  = torch.load('m.mel').to(device)
    # x = torch.FloatTensor(x).to(device)
    y_g_hat = generator(x)
    audio = y_g_hat.squeeze()
    # print(audio.shape)
    # print(audio[:100])
    # print(torch.sum(audio))
    with open('torch.p', 'wb') as f:
      pickle.dump(audio.numpy(), f)

    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    output_file = 'rand_generated_e2e.wav'
    write(output_file, h.sampling_rate, audio)
    print(output_file)


def main():
  print('Initializing Inference Process..')

  parser = argparse.ArgumentParser()
  parser.add_argument('--input_mels_dir', default='test_mel_files')
  parser.add_argument('--output_dir', default='generated_files_from_mel')
  parser.add_argument('--checkpoint_file', required=True)
  a = parser.parse_args()

  config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
  with open(config_file) as f:
    data = f.read()

  global h
  json_config = json.loads(data)
  h = AttrDict(json_config)

  torch.manual_seed(h.seed)
  global device
  if torch.cuda.is_available():
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  inference(a)


if __name__ == '__main__':
  main()
