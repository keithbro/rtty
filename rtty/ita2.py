class Ita2():
  MODE_LETTERS = 'L'
  MODE_FIGURES = 'F'
  SHIFT_TO_LETTERS = 'LS'
  SHIFT_TO_FIGURES = 'FS'

  # https://www.dcode.fr/baudot-code
  ITA2 = {
    MODE_LETTERS: [
      "", "E", "\n", "A", " ", "S", "I", "U", "", "D", "R", "J", "N", "F", "C", "K", "T", "Z", "L", "W", "H",
      "Y", "P", "Q", "O", "B", "G", SHIFT_TO_FIGURES, "M", "X", "V", SHIFT_TO_LETTERS,
    ],
    MODE_FIGURES: [
      "", "3", "\n", "-", " ", "🔔", "8", "7", "", "$", "4", "'", ",", "!", ":", "(", "5", "\"", ")", "2", "#",
      "6", "0", "1", "9", "?", "&", SHIFT_TO_FIGURES, ".", "/", ";", SHIFT_TO_LETTERS,
    ],
  }

  def __init__(self, reverse_bits, logger):
    self.reverse_bits = reverse_bits
    self.logger = logger
    self.mode = self.MODE_LETTERS

  def decode(self, data_bits):
    self.logger.debug("Data bits: " + str(data_bits))

    if self.reverse_bits:
      data_bits = reversed(data_bits)

    idx = int("".join(map(str, data_bits)), 2)
    symbol = self.ITA2[self.mode][idx]

    if symbol == self.SHIFT_TO_FIGURES:
      self.mode = self.MODE_FIGURES
      return
    elif symbol == self.SHIFT_TO_LETTERS:
      self.mode = self.MODE_LETTERS
      return

    return symbol
