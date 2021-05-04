from argparse import ArgumentParser
from pathlib import Path

import soundfile as sf

from .waveRNN.mel2wave import mel2wave

parser = ArgumentParser()
parser.add_argument('--text', type=str)
parser.add_argument('--output', default='clip.wav', type=Path)
parser.add_argument('--sample-rate', default=16000, type=int)
parser.add_argument('--use-nat', default=False, action='store_true')
args = parser.parse_args()

if args.use_nat:
  from .nat.text2mel import text2mel
  mel = text2mel(args.text)
else:
  from .tacotron.text2mel import text2mel
  mel = text2mel(args.text)
wave = mel2wave(mel)

print('writing output to file', args.output)
sf.write(str(args.output), wave, samplerate=args.sample_rate)
