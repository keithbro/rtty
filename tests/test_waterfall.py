from rtty.waterfall import Waterfall


def test_accumulates_snapshots():
  """update() stores each FFT snapshot as a row in the spectrogram data."""
  wf = Waterfall(max_freq=4000)
  wf.update({100: 5.0, 200: 10.0, 300: 2.0})
  wf.update({100: 3.0, 200: 8.0, 300: 6.0})
  wf.update({100: 1.0, 200: 4.0, 300: 9.0})

  assert len(wf.rows) == 3
  # Each row should be a list/array with consistent length
  assert len(wf.rows[0]) == len(wf.rows[1]) == len(wf.rows[2])


def test_clamps_to_frequency_range():
  """Frequencies outside [0, max_freq) are ignored; rows have consistent width."""
  wf = Waterfall(max_freq=100)
  wf.update({50: 5.0, 150: 99.0, -10: 3.0})

  assert len(wf.rows) == 1
  row = wf.rows[0]
  assert len(row) == 100
  assert row[50] == 5.0
  # Out-of-range freqs should not appear
  assert sum(row) == 5.0
