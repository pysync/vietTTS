from .config import DurationInput

import numpy as np
def text_mel_data_loader(data_dir, batch_size, mode):
  assert mode in ['train', 'val']
  p = np.zeros((batch_size, 10, dtype=np.int32)
  l = np.zeros((batch_size, 10, dtype=np.int32)
  while True:

