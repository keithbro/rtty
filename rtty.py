
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

# Tested on:
# 4582.97KHz
# http://meinsdr.ddns.net:8073/

SAMPLE_RATE = 8000
BAUD = 50
SHIFT = 450 # Hz
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
    "01000": "", # \r
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
    "01000": "", # \r
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

def decode_chunk(chunk, frequencies, confidences):
  decoded = np.frombuffer(chunk, dtype=np.int16)
  bins = np.fft.fft(decoded)
  freqs = np.fft.fftfreq(len(bins))

  freq_amplitudes = {}
  for idx, bin in enumerate(bins):
    freq = int(abs(freqs[idx] * SAMPLE_RATE))
    amp = np.abs(bin)
    if freq_amplitudes.get(freq):
      freq_amplitudes[freq] = freq_amplitudes.get(freq) + amp
    else:
      freq_amplitudes[freq] = amp

  sorted_by_amp = {
    k: v for k, v in sorted(freq_amplitudes.items(), key=lambda item: item[1], reverse=True)
  }
  logger.debug(sorted_by_amp)
  max_amp = list(sorted_by_amp.values())[0]
  dominant_freq = list(sorted_by_amp.keys())[0]
 
  if len(frequencies) > 50:
    frequencies.pop(0)

  if len(confidences) > 50:
    confidences = [np.average(confidences)]

  frequencies.append(dominant_freq)
  # logger.debug(frequencies)
  average_freq = np.average(frequencies)
  diff = dominant_freq - average_freq
  if diff > 0:
    inactive_freq = dominant_freq - SHIFT
  else:
    inactive_freq = dominant_freq + SHIFT

  logger.debug("Highest freq: "     + str(np.max(frequencies)) + 'Hz')
  logger.debug("Lowest freq: "      + str(np.min(frequencies)) + 'Hz')
  logger.debug("Average dominant freq: " + str(average_freq) + 'Hz')
  logger.debug("Dominant freq: "    + str(dominant_freq) + 'Hz')
  logger.debug("Inactive freq: "    + str(inactive_freq) + 'Hz')

  confidence_freq = abs(diff)
  confidence = confidence_freq / (SHIFT / 2) * 100
  confidences.append(confidence)

  logger.debug("Confidence freq: " + str(confidence_freq) + "Hz")
  logger.debug("Confidence: " + str(int(confidence)) + "%")
  logger.debug("Average Confidence: " + str(np.average(confidences)) + '%')

  if diff > 0:
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
      # logger.info("Symbol: \"" + symbol + "\"")
      False or print(symbol, end="", flush=True)
  except:
    print("ERROR: mode: " + mode + ", binary: " + symbol_bits)

  return [], mode
  
bits = []
mode = "L"
frequencies = []
confidences = []

while True:
  logger.debug("Fetching another bit...")
  if len(bits) == 6:
    # logger.debug("Expecting a stop bit.")
    chunk_size = int(CHUNK_SIZE * 1.5)
  else:
    chunk_size = CHUNK_SIZE
  
  logger.debug("Reading chunk of size: " + str(chunk_size))
  chunk = input_stream.read(chunk_size)
  # print(chunk)
  output_stream.write(chunk)

  bit = decode_chunk(chunk, frequencies, confidences)
  # print(bit, end="")
  bits.append(bit)
  bits, mode = xxx(bits, mode)