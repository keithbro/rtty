
import numpy as np
import pyaudio
import logging
import rtty

logger = logging.getLogger('rtty')
logger.setLevel(logging.DEBUG)

ch = logging.FileHandler('debug.log')
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger.addHandler(ch)

SAMPLE_RATE = 8000
BAUD = 50
SHIFT = 450 # Hz
STOP_BITS = 1.5
CHUNK_SIZE = int(SAMPLE_RATE / BAUD) # 61
REVERSE_BITS = True
INVERTED = True

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

signal_decoder = rtty.Decoder(SAMPLE_RATE, BAUD, SHIFT, STOP_BITS, INVERTED, logger)
ita2_decoder = rtty.Ita2(REVERSE_BITS, logger)

signal_decoder.synchronise(input_stream)

while True:
  chunk_size = signal_decoder.chunk_size()

  chunk = input_stream.read(chunk_size)
  output_stream.write(chunk)

  bit = signal_decoder.decode(chunk)

  if bit is None:
    continue

  data_bits = signal_decoder.process(bit)

  if len(data_bits) == 0:
    continue

  symbol = ita2_decoder.decode(data_bits)

  if symbol:
    print(symbol, end="", flush=True)
