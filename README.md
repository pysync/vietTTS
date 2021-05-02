A Vietnamese TTS
================

Tacotron + WaveRNN for vietnamese dataset.

Install
-------


```sh
git clone https://github.com/NTT123/vietTTS.git
cd vietTTS 
pip3 install -e .
```

Download reinfo dataset
-----------------------

```sh
bash ./scripts/download_reinfo_dataset.sh
```


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


Text to melspectrogram

```sh
python3 -m vietTTS.nat.text2mel --text "hôm qua em tới trường" --output mel.png
```

Train Tacotron 
--------------

```sh
python3 -m vietTTS.tacotron.trainer
```

Train waveRNN
-------------

```sh
python3 -m vietTTS.waveRNN.trainer
```


Synthesize speech
-----------------

```sh
python3 -m vietTTS.synthesizer --text="####### hôm qua em tới trường #######" --output=clip.wav
```