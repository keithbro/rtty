import wave


class WavStream:
  def __init__(self, path):
    self._wav = wave.open(path, 'rb')

    if self._wav.getnchannels() != 1:
      raise ValueError(f"Expected mono audio, got {self._wav.getnchannels()} channels")

    if self._wav.getsampwidth() != 2:
      raise ValueError(f"Expected 16-bit audio, got {self._wav.getsampwidth() * 8}-bit")

    self.sample_rate = self._wav.getframerate()

  def read(self, num_samples):
    return self._wav.readframes(num_samples)

  def close(self):
    self._wav.close()
