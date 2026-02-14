import logging

import numpy as np
import pytest

from rtty import Decoder

logger = logging.getLogger('test')

SAMPLE_RATE = 8000
BAUD = 50
SHIFT = 450
STOP_BITS = 1.5


def make_decoder():
  return Decoder(SAMPLE_RATE, BAUD, SHIFT, STOP_BITS, inverted=True, logger=logger)


def test_process_valid_frame():
  decoder = make_decoder()
  # Start bit (0), 5 data bits, stop bit (1)
  bits = [0, 1, 0, 0, 0, 1, 1]
  results = []
  for bit in bits:
    result = decoder.process(bit)
    results.append(result)

  # Only the last bit should produce output
  for r in results[:-1]:
    assert r == []
  assert results[-1] == [1, 0, 0, 0, 1]


def test_process_rejects_invalid_start_bit():
  decoder = make_decoder()
  # First bit is 1, not a valid start bit (should be 0)
  result = decoder.process(1)
  assert result == []


def test_process_rejects_invalid_stop_bit():
  decoder = make_decoder()
  # Start bit (0), 5 data bits, invalid stop bit (0)
  bits = [0, 1, 0, 0, 0, 1, 0]
  results = []
  for bit in bits:
    results.append(decoder.process(bit))

  # Last result should be empty (invalid stop bit)
  assert results[-1] == []


def test_process_resets_after_frame():
  decoder = make_decoder()
  # Send a valid frame
  for bit in [0, 1, 0, 0, 0, 1, 1]:
    decoder.process(bit)

  # Bits should be reset, next frame starts fresh
  assert decoder.bits == []


def test_chunk_size_normal():
  decoder = make_decoder()
  # With no bits accumulated, chunk size is sample_rate / baud
  assert decoder.chunk_size() == int(SAMPLE_RATE / BAUD)


def test_chunk_size_stop_bit():
  decoder = make_decoder()
  # Simulate 6 bits accumulated (about to read stop bit)
  decoder.bits = [0, 1, 0, 0, 0, 1]
  expected = int((SAMPLE_RATE / BAUD) * STOP_BITS)
  assert decoder.chunk_size() == expected


class NoiseStream:
  """Stream that returns random noise, never producing a clear RTTY signal."""
  def read(self, num_samples):
    samples = np.random.randint(-100, 100, size=num_samples, dtype=np.int16)
    return samples.tobytes()


class EofStream:
  """Stream that returns EOF immediately."""
  def read(self, num_samples):
    return b''


def test_synchronise_timeout():
  decoder = make_decoder()
  with pytest.raises(TimeoutError, match="Synchronisation timed out"):
    decoder.synchronise(NoiseStream(), timeout=0.1)


def test_on_fft_callback_invoked():
  """decode() calls the on_fft callback with a freq_amplitudes dict."""
  received = []
  def on_fft(freq_amplitudes):
    received.append(freq_amplitudes)

  decoder = Decoder(SAMPLE_RATE, BAUD, SHIFT, STOP_BITS, inverted=True, logger=logger, on_fft=on_fft)
  # Generate a 1kHz tone chunk
  t = np.arange(int(SAMPLE_RATE / BAUD)) / SAMPLE_RATE
  tone = (np.sin(2 * np.pi * 1000 * t) * 10000).astype(np.int16)
  decoder.decode(tone.tobytes())

  assert len(received) == 1
  assert isinstance(received[0], dict)
  # Should contain integer freq keys and float amplitude values
  for freq, amp in received[0].items():
    assert isinstance(freq, int)
    assert isinstance(amp, float) or isinstance(amp, np.floating)


def test_synchronise_no_timeout():
  decoder = make_decoder()
  # With timeout=None and a stream that hits EOF, should return without raising.
  decoder.synchronise(EofStream(), timeout=None)
