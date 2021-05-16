A Vietnamese TTS
================

Tacotron + HiFiGAN vocoder for vietnamese datasets.

A synthesized audio clip is at [assets/infore/clip.wav](assets/infore/clip.wav).

Install
-------


```sh
git clone https://github.com/NTT123/vietTTS.git
cd vietTTS 
pip3 install -e .
```


Quick start using pretrained models
----------------------------------
```sh
bash ./scripts/quick_start.sh
```


Download InfoRe dataset
-----------------------

```sh
bash ./scripts/download_aligned_infore_dataset.sh
```

**Note**: this is a denoised and aligned version of the original dataset which is donated by the InfoRe Technology company (see [here](https://www.facebook.com/groups/j2team.community/permalink/1010834009248719/)). You can download the original dataset (**InfoRe Technology 1**) at [here](https://github.com/TensorSpeech/TensorFlowASR/blob/main/README.md#vietnamese).


Train duration model
--------------------

```sh
python3 -m vietTTS.nat.duration_trainer
```


Train acoustic model
--------------------
```sh
python3 -m vietTTS.nat.acoustic_trainer
```



Train HiFiGAN
-------------

We use the original implementation from HiFiGAN authors at https://github.com/jik876/hifi-gan. Use the config file at `assets/hifigan/config.json` to train your model.

Then, use the following command to convert pytorch model to haiku format:
```sh
python3 -m vietTTS.hifigan.convert_torch_model_to_haiku \
  --config-file=assets/hifigan/config.json \
  --checkpoint-file=/path/to/pytorch/model/g_[latest_checkpoint]
```

Synthesize speech
-----------------

```sh
python3 -m vietTTS.synthesizer \
  --use-nat \
  --use-hifigan \
  --lexicon-file=train_data/lexicon.txt 
  --text="hôm qua em tới trường" \
  --output=clip.wav
```