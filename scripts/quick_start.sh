if [ ! -f assets/infore/hifigan/g_00430000 ]; then
  pip3 install gdown
  echo "Downloading models..."
  mkdir -p -p assets/infore/{nat,hifigan}
  gdown --id 1-uLp0KBu1F5fHPREYg63qCkD9cAup9Mq -O assets/infore/nat/duration_ckpt_latest.pickle
  gdown --id 10P7Q-hwZIb74PD8ZiE-7PghbzXgbCbJN -O assets/infore/nat/acoustic_ckpt_latest.pickle
  gdown --id 10sQsFtDbJiumNxJ4EJkhQafRbzL9uIk_ -O assets/infore/hifigan/g_00430000
  python3 -m vietTTS.hifigan.convert_torch_model_to_haiku --config-file=assets/hifigan/config.json --checkpoint-file=assets/infore/hifigan/g_00430000
fi

echo "Generate audio clip"
text=`cat assets/transcript.txt`
python3 -m vietTTS.synthesizer --text "$text" --output assets/infore/clip.wav --use-hifigan --use-nat --lexicon-file assets/infore/lexicon.txt --silence-duration 0.2