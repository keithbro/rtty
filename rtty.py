
import numpy as np
import pyaudio
import pprint
import sys
import re
import time

# np.set_printoptions(threshold=sys.maxsize)

pp = pprint.PrettyPrinter(indent=4)

message = "CQ AB1HL FN42"
ascii = message.encode('ascii')

SAMPLE_RATE = 8000
SECONDS = 3
BAUD = 45.45
# BIT_RATE = BAUD * 8 # 364
CHUNK_SIZE = int(SAMPLE_RATE / BAUD) # 61
f = 490
volume = 20000

samples = np.linspace(0, SECONDS, int(SAMPLE_RATE * SECONDS), endpoint=False)
signal = np.sin(2 * np.pi * f * samples) * volume
signal = np.int16(signal)

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

def decode_to_bit(chunk):
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
  freq_in_hertz = abs(freq * SAMPLE_RATE)
  # print(freq_in_hertz)
  if freq_in_hertz > 950:
    return "1"
  else:
    return "0"
  
bits = []

for i in range(0, int(BAUD * SECONDS)):
  chunk = input_stream.read(CHUNK_SIZE)
  bits.append(decode_to_bit(chunk))
  decoded = np.frombuffer(chunk, dtype=np.int16)
  signal2 = np.append(signal2, decoded)

output_stream.write(signal2)

print("".join(bits))

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
    "00010": " ",
    "00100": " ",
    "00011": "-",
    "11010": "&",
    "10100": "#",
    "11100": ".",
    "01100": ",",
    "11110": ";",
    "01110": ":",
    "01111": "{",
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
    "11111": "LS",
    "00010": "\n",
    "01000": "\n", # \r
  }
}

mode = "L"
pos = 0

def fs(mode):
  if mode == "L":
    return "F"
  elif mode == "F":
    return "L"

while pos < len(bits):
  c = bits[pos:pos+7]
  bit_str = "".join(c)
  x = re.match(r'^0[0,1]{5}1$', bit_str)

  if x is None:
    pos += 1
    continue

  char = bit_str[1:6]
  if True:
    char = "".join(reversed(char))
  
  pos += 7

  try:
    ddd = baudot[mode][char]

    if ddd == "FS":
      mode = "F"
    elif ddd == "LS":
      mode = "L"
    else:
      print(ddd, end="")
  except:
    print("ERROR: mode: " + mode + ", binary: " + char)

input_stream.stop_stream()
input_stream.close()