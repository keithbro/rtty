import numpy as np


class Waterfall:
  """Accumulates FFT snapshots for spectrogram visualization."""

  def __init__(self, max_freq):
    # Frequency bins span 0 to max_freq (1 Hz resolution).
    self.max_freq = max_freq
    self.rows = []

  def update(self, freq_amplitudes):
    """Append one FFT snapshot as a row in the spectrogram."""
    row = np.zeros(self.max_freq)
    for freq, amp in freq_amplitudes.items():
      if 0 <= freq < self.max_freq:
        row[freq] = float(amp)
    self.rows.append(row)

  def show(self):
    """Render the accumulated spectrogram using matplotlib."""
    import matplotlib.pyplot as plt

    if not self.rows:
      return

    data = np.array(self.rows)
    plt.figure(figsize=(12, 6))
    plt.imshow(
      data.T,
      aspect='auto',
      origin='lower',
      extent=[0, len(self.rows), 0, self.max_freq],
    )
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time (chunks)')
    plt.ylabel('Frequency (Hz)')
    plt.title('RTTY Spectrogram')
    plt.tight_layout()
    plt.show()
