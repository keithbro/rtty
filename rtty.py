
import numpy as np
import pyaudio
import pprint
import sys
import re
import time
import logging

logger = logging.getLogger('rtty')
logger.setLevel(logging.DEBUG)

ch = logging.FileHandler('debug.log')
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger.addHandler(ch)

SAMPLE_RATE = 8000
BAUD = 50
CHUNK_SIZE = int(SAMPLE_RATE / BAUD) # 61
REVERSE_BITS = True

# https://www.dcode.fr/baudot-code
baudot = {
  "L": {
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
    "11011": "FS", #
    "11111": " ", #
    "00100": " ", #
    "00010": "\n",
    "01000": "\n", # \r
  },
  "F": {
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
    "11111": "LS",
    "00010": "\n",
    "01000": "\n", # \r
  }
}

p = pyaudio.PyAudio()

input_stream = p.open(rate=SAMPLE_RATE,
                      channels=1,
                      format=pyaudio.paInt16,
                      input_device_index=4,
                      input=True)

output_stream = p.open(rate=SAMPLE_RATE,
                       channels=1,
                       format=pyaudio.paInt16,
                       output_device_index=1,
                       output=True)

signal2 = np.ndarray(shape=(1,0), dtype=np.int16)

def decode_chunk(chunk, frequencies):
  decoded = np.frombuffer(chunk, dtype=np.int16)
  bins = np.fft.fft(decoded)
  freqs = np.fft.fftfreq(len(bins))

  x = []
  for idx, bin in enumerate(bins):
    freq = int(abs(freqs[idx] * SAMPLE_RATE))
    amp = np.abs(bin)
    x.append((freq, amp))

  sorted_by_second = sorted(x, key=lambda tup: tup[1])

  # print(sorted_by_second)
  idx = np.argmax(np.abs(bins))
  freq = freqs[idx]
  freq_in_hertz = int(abs(freq * SAMPLE_RATE))
  logger.debug("Dominant frequency: " + str(freq_in_hertz) + 'Hz')

  if len(frequencies) > 50:
    frequencies.pop(0)

  frequencies.append(freq_in_hertz)
  logger.debug(frequencies)
  average_freq = np.average(frequencies)
  logger.debug("Average dominant freq: " + str(average_freq))

  if freq_in_hertz > average_freq:
    return "0"
  else:
    return "1"

def fs(mode):
  if mode == "L":
    return "F"
  elif mode == "F":
    return "L"

def xxx(bits, mode):
  logger.debug(bits)
  bit_str = "".join(bits)
  # print(bit_str)

  if bits[0] == "1":
    return [], mode

  if(len(bits) < 7):
    return bits, mode

  if re.match(r'^0(0|1){5}1$', bit_str) is None:
    return bits[1:], mode

  #print(bit_str)
  symbol_bits = bit_str[1:6]
  #print(symbol_bits)
  if REVERSE_BITS:
    symbol_bits = "".join(reversed(symbol_bits))

  try:
    symbol = baudot[mode][symbol_bits]

    if symbol == "FS":
      mode = "F"
    elif symbol == "LS":
      mode = "L"
    else:
      logger.info("Symbol: \"" + symbol + "\"")
      False or print(symbol, end="", flush=True)
  except:
    print("ERROR: mode: " + mode + ", binary: " + symbol_bits)

  return [], mode
  
bits = []
mode = "L"
frequencies = []

while True:
  logger.debug("Fetching another bit...")
  if len(bits) == 6:
    logger.debug("Expecting a stop bit.")
    chunk_size = int(CHUNK_SIZE * 1.5)
  else:
    chunk_size = CHUNK_SIZE
  
  logger.debug("Reading chunk of size: " + str(chunk_size))
  chunk = input_stream.read(chunk_size)
  # print(chunk)
  output_stream.write(chunk)

  bit = decode_chunk(chunk, frequencies)
  # print(bit, end="")
  bits.append(bit)
  bits, mode = xxx(bits, mode)