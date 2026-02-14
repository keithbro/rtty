import logging
from rtty import Ita2

logger = logging.getLogger('test')


def test_decode_letter_e():
  ita2 = Ita2(reverse_bits=False, logger=logger)
  assert ita2.decode([0, 0, 0, 0, 1]) == "E"


def test_decode_letter_a():
  ita2 = Ita2(reverse_bits=False, logger=logger)
  assert ita2.decode([0, 0, 0, 1, 1]) == "A"


def test_decode_letter_t():
  ita2 = Ita2(reverse_bits=False, logger=logger)
  assert ita2.decode([1, 0, 0, 0, 0]) == "T"


def test_decode_space():
  ita2 = Ita2(reverse_bits=False, logger=logger)
  assert ita2.decode([0, 0, 1, 0, 0]) == " "


def test_decode_newline():
  ita2 = Ita2(reverse_bits=False, logger=logger)
  assert ita2.decode([0, 0, 0, 1, 0]) == "\n"


def test_figure_shift():
  ita2 = Ita2(reverse_bits=False, logger=logger)
  # Shift to figures (index 27 = 11011)
  result = ita2.decode([1, 1, 0, 1, 1])
  assert result is None
  assert ita2.mode == Ita2.MODE_FIGURES
  # Now decode index 1 (00001) which is "3" in figures mode
  assert ita2.decode([0, 0, 0, 0, 1]) == "3"


def test_letter_shift():
  ita2 = Ita2(reverse_bits=False, logger=logger)
  # Switch to figures first
  ita2.decode([1, 1, 0, 1, 1])
  assert ita2.mode == Ita2.MODE_FIGURES
  # Shift back to letters (index 31 = 11111)
  result = ita2.decode([1, 1, 1, 1, 1])
  assert result is None
  assert ita2.mode == Ita2.MODE_LETTERS
  # Now decode should give letters
  assert ita2.decode([0, 0, 0, 0, 1]) == "E"


def test_reverse_bits():
  ita2 = Ita2(reverse_bits=True, logger=logger)
  # E is index 1 = 00001, reversed input should be 10000
  assert ita2.decode([1, 0, 0, 0, 0]) == "E"


def test_reverse_bits_letter_t():
  ita2 = Ita2(reverse_bits=True, logger=logger)
  # T is index 16 = 10000, reversed input should be 00001
  assert ita2.decode([0, 0, 0, 0, 1]) == "T"
