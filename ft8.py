
import numpy as np
import pyaudio
import pprint
import sys

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

def decode(data):
  print("OK")

# 8000 / 1024 * 1 = 7.8 = 7
# 8000 samples / 45.45 BAUD * SECONDS * 1 = 176 reads

for i in range(0, int(SAMPLE_RATE / BAUD * SECONDS)):
    data = input_stream.read(CHUNK)
    frequencies = decode(data)
    print(len(data))
    #decoded = np.frombuffer(data, dtype=np.int16)
    #signal2 = np.append(signal2, decoded)

exit()

input_stream.stop_stream()
input_stream.close()

print(signal2)
output_stream.write(signal2)

bins = np.fft.fft(signal2)
# print(bins)
print(len(bins))

freqs = np.fft.fftfreq(len(bins))
# print(freqs)

x = []
for idx, bin in enumerate(bins):
  freq = int(abs(freqs[idx] * SAMPLE_RATE))
  amp = np.abs(bin)
  x.append((freq, amp))

sorted_by_second = sorted(x, key=lambda tup: tup[1])

print(sorted_by_second)

idx = np.argmax(np.abs(bins))
freq = freqs[idx]
print(freq)

freq_in_hertz = abs(freq * SAMPLE_RATE)
print(freq_in_hertz)