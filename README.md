# RTTY Decoder

A learning experiment, written in Python.

Tested on: 4582.97KHz at http://meinsdr.ddns.net:8073/
And also with https://www.sigidwiki.com/wiki/RTTY

## Getting Started

Install PortAudio (required by PyAudio):

```bash
brew install portaudio
```

Install Python dependencies:

```bash
pip3 install -r requirements.txt
```

Decode a WAV file:

```bash
python3 decode_wav.py --skip-sync --no-invert tests/fixtures/rtty_450hz_50bd.wav
```

Decode with a spectrogram visualization:

```bash
python3 decode_wav.py --skip-sync --no-invert --waterfall tests/fixtures/rtty_450hz_50bd.wav
```