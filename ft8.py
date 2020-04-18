
import numpy as np
import pyaudio
import pprint
import sys
import re

# np.set_printoptions(threshold=sys.maxsize)

pp = pprint.PrettyPrinter(indent=4)

message = "CQ AB1HL FN42"
ascii = message.encode('ascii')

# CHUNK = 1024
SAMPLE_RATE = 8000
SECONDS = 1
BAUD = 45.45
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

CHUNK = int(SAMPLE_RATE/BAUD)

def decode(data):
  decoded = np.frombuffer(data, dtype=np.int16)
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
  if freq_in_hertz > 950:
    return "1"
  else:
    return "0"
  

# 8000 / 1024 * 1 = 7.8 = 7
# 8000 samples / 45.45 BAUD * SECONDS * 1 = 176 reads

res = []

for i in range(0, int(SAMPLE_RATE / BAUD * SECONDS)):
    data = input_stream.read(CHUNK)
    res.append(decode(data))
    decoded = np.frombuffer(data, dtype=np.int16)
    signal2 = np.append(signal2, decoded)

print("".join(res))

# https://www.dcode.fr/baudot-code
baudot = {
  "L": {
    "10000": "A",
    "00110": "B",
    "01110": "C", #
    "01000": "E",
    "01101": "F", #
    "11010": "G", #
    "10100": "H",
    "01111": "K", #
    "10010": "L",
    "01011": "M",
    "01100": "N", #
    "11100": "O",
    "10110": "P", #
    "10111": "Q",
    "01010": "R", #
    "00101": "S",
    "10101": "T",
    "00111": "U", #
    "11110": "V",
    "": "W",
    "01001": "X",
    "00100": "Y",
    "11001": "Z",
    "11000": "/",
    "10001": "-",
    "00010": "FS",
    "11111": " ",
  },
  "F": {
    "00010": " ",
    "11010": "&",
    "10100": "#",
    "11100": ".",
    "11110": ";",
    "10110": "0",
    "01010": "4",
    "00111": "7",
    "11000": "9",
    "11111": "LS",
  }
}

baudot2 = {
  "L": {
    "11000": "A",
    "10011": "B",
    "10110": "F",
    "01001": "L",
    "11100": "U",
    "01111": "V",
    "10001": "Z",
  }
}

mode = "L"
pos = 0
lock = False

def fs(mode):
  if mode == "L":
    return "F"
  elif mode == "F":
    return "L"

while pos < len(res):
  c = res[pos:pos+5]
  bit_str = "".join(c)
  x = re.match(r'^0.+11$', bit_str)

  if x is None:
    pos += 1
    next

  lock = True
  pos += 5

  try:
    decoded = baudot[mode][bit_str]
  except:
    print("ERROR")
    print("mode: " + mode)
    print("binary: " + bit_str)

  if decoded == "FS":
    mode = "F"
  elif decoded == "LS":
    mode = "L"
  else:
    print(decoded)

input_stream.stop_stream()
input_stream.close()
output_stream.write(signal2)