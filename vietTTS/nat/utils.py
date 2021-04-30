import pickle
from pathlib import Path


def load_latest_ckpt(ckpt_dir: Path):
  with open(ckpt_dir/'latest_state.pickle', 'wb') as f:
    dic = pickle.load(f, fix_imports=True)
  return dic['step'], dic['params'], dic['aux'], dic['rng'], dic['optim_state']


def save_ckpt(step, params, aux, rng, optim_state, ckpt_dir: Path):
  dic = {'step': step, 'params': params, 'aux': aux, 'rng': rng, 'optim_state': optim_state}
  with open(ckpt_dir/'latest_state.pickle', 'wb') as f:
    pickle.dump(dic, f, fix_imports=True)
