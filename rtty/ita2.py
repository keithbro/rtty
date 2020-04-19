class Ita2():
  MODE_LETTERS = 'L'
  MODE_FIGURES = 'F'
  SHIFT_TO_LETTERS = 'LS'
  SHIFT_TO_FIGURES = 'FS'

  # https://www.dcode.fr/baudot-code
  ITA2 = {
    MODE_LETTERS: {
      "00000": "", # NULL
      "00011": "A", #
      "11001": "B", #
      "01110": "C", #
      "01001": "D", #
      "00001": "E", #
      "01101": "F", #
      "11010": "G", #
      "10100": "H", #
      "00110": "I", #
      "01011": "J", #
      "01111": "K", #
      "10010": "L", #
      "11100": "M", #
      "01100": "N", #
      "11000": "O", #
      "10110": "P", #
      "10111": "Q", #
      "01010": "R", #
      "00101": "S", #
      "10000": "T", #
      "00111": "U", #
      "11110": "V", #
      "10011": "W", #
      "11101": "X", #
      "10101": "Y", #
      "10001": "Z", #
      "11011": SHIFT_TO_FIGURES,
      "11111": " ", #
      "00100": " ", #
      "00010": "\n",
      "01000": "", # \r
    },
    MODE_FIGURES: {
      "00000": "", # NULL
      "00010": " ",
      "00100": " ",
      "00011": "-",
      "11010": "&",
      "10100": "#",
      "01001": "$",
      "11100": ".",
      "01100": ",",
      "11110": ";",
      "01011": "'",
      "01110": ":",
      "01111": "{",
      "10010": ")",
      "10001": "\"",
      "11101": "/",
      "11001": "?",
      "01101": "!",
      "10110": "0",
      "10111": "1",
      "10011": "2",
      "00001": "3",
      "01010": "4",
      "10000": "5",
      "10101": "6",
      "00111": "7",
      "00110": "8",
      "11000": "9",
      "00101": "BELL",
      "11011": " ",
      "11111": SHIFT_TO_LETTERS,
      "00010": "\n",
      "01000": "", # \r
    }
  }

  def __init__(self, reverse_bits, logger):
    self.reverse_bits = reverse_bits
    self.logger = logger
    self.mode = self.MODE_LETTERS

  def decode(self, data_bits):
    self.logger.debug(data_bits)

    if self.reverse_bits:
      data_bits = reversed(data_bits)

    symbol = self.ITA2[self.mode]["".join(map(str, data_bits))]

    if symbol == self.SHIFT_TO_FIGURES:
      self.mode = self.MODE_FIGURES
      return
    elif symbol == self.SHIFT_TO_LETTERS:
      self.mode = self.MODE_LETTERS
      return

    return symbol