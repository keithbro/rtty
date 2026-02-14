import logging
import os
from rtty import Decoder, Ita2, WavStream

logger = logging.getLogger('test')

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
WAV_FILE = os.path.join(FIXTURE_DIR, 'rtty_450hz_50bd.wav')


def decode_wav(wav_path, inverted=False, reverse_bits=True):
  """Decode an RTTY WAV file and return the decoded text."""
  stream = WavStream(wav_path)
  decoder = Decoder(
    stream.sample_rate, baud=50, shift=450, stop_bits=1.5,
    inverted=inverted, logger=logger,
  )
  ita2 = Ita2(reverse_bits=reverse_bits, logger=logger)

  output = []
  while True:
    chunk_size = decoder.chunk_size()
    chunk = stream.read(chunk_size)

    if len(chunk) < chunk_size * 2:
      break

    bit = decoder.decode(chunk)
    if bit is None:
      continue

    data_bits = decoder.process(bit)
    if len(data_bits) == 0:
      continue

    symbol = ita2.decode(data_bits)
    if symbol:
      output.append(symbol)

  stream.close()
  return ''.join(output)


def test_decode_fox_message():
  """Integration test: decode the sigidwiki RTTY sample end-to-end."""
  text = decode_wav(WAV_FILE, inverted=False, reverse_bits=True)
  assert "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG" in text


def test_decode_contains_digits():
  """The sample also contains digits 0123456789."""
  text = decode_wav(WAV_FILE, inverted=False, reverse_bits=True)
  assert "0123456789" in text


def test_wav_stream_properties():
  """WavStream exposes correct sample rate."""
  stream = WavStream(WAV_FILE)
  assert stream.sample_rate == 8000
  stream.close()


def test_wav_stream_read_returns_bytes():
  stream = WavStream(WAV_FILE)
  data = stream.read(160)
  assert isinstance(data, bytes)
  assert len(data) == 160 * 2  # 16-bit = 2 bytes per sample
  stream.close()


def test_wav_stream_eof():
  """Reading past EOF returns empty bytes."""
  stream = WavStream(WAV_FILE)
  # Read all frames
  stream.read(100000)
  # Next read should return empty
  data = stream.read(160)
  assert data == b''
  stream.close()
