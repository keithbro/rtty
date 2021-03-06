import re
import numpy as np
from random import randrange

class Decoder:
  START_BIT = 0
  STOP_BIT = 1

  def __init__(self, sample_rate, baud, shift, stop_bits, inverted, logger):
    self.frequencies = []
    self.confidences = []
    self.bits = []
    self.sample_rate = sample_rate
    self.baud = baud
    self.shift = shift
    self.logger = logger
    self.stop_bits = stop_bits
    self.inverted = inverted

  def chunk_size(self):
    chunk_size = self.sample_rate / self.baud

    if len(self.bits) == 6:
      return int(chunk_size * self.stop_bits)
    else:
      return int(chunk_size)

  def decode(self, chunk):
    # self.logger.debug(chunk)
    if int.from_bytes(chunk, "big") == 0:
      return

    decoded = np.frombuffer(chunk, dtype=np.int16)
    bins = np.fft.fft(decoded)
    freqs = np.fft.fftfreq(len(bins))

    freq_amplitudes = {}
    for idx, bin in enumerate(bins):
      freq = int(abs(freqs[idx] * self.sample_rate))
      amp = np.abs(bin)
      if freq_amplitudes.get(freq):
        freq_amplitudes[freq] = freq_amplitudes.get(freq) + amp
      else:
        freq_amplitudes[freq] = amp

    sorted_by_amp = {
      k: v for k, v in sorted(freq_amplitudes.items(), key=lambda item: item[1], reverse=True)
    }
    # self.logger.debug(sorted_by_amp)
    max_amp = list(sorted_by_amp.values())[0]
    dominant_freq = list(sorted_by_amp.keys())[0]
  
    if len(self.frequencies) > 50:
      self.frequencies.pop(0)

    if len(self.confidences) > 50:
      self.confidences = [np.average(self.confidences)]

    self.frequencies.append(dominant_freq)
    # logger.debug(frequencies)
    average_freq = np.average(self.frequencies)
    diff = dominant_freq - average_freq
    if diff > 0:
      inactive_freq = dominant_freq - self.shift
    else:
      inactive_freq = dominant_freq + self.shift

    self.logger.debug("Highest freq: "     + str(np.max(self.frequencies)) + 'Hz')
    self.logger.debug("Lowest freq: "      + str(np.min(self.frequencies)) + 'Hz')
    self.logger.debug("Average dominant freq: " + str(average_freq) + 'Hz')
    self.logger.debug("Dominant freq: "    + str(dominant_freq) + 'Hz')
    self.logger.debug("Inactive freq: "    + str(inactive_freq) + 'Hz')

    confidence_freq = abs(diff)
    confidence = confidence_freq / (self.shift / 2) * 100
    self.confidences.append(confidence)

    self.logger.debug("Confidence freq: " + str(confidence_freq) + "Hz")
    self.logger.debug("Confidence: " + str(int(confidence)) + "%")
    self.logger.debug("Average Confidence: " + str(np.average(self.confidences)) + '%')

    if diff < 0 and self.inverted or diff > 0 and not self.inverted:
      bit = 1
    else:
      bit = 0

    self.logger.debug("Bit: " + str(bit))

    return bit

  def process(self, bit):
    self.bits.append(bit)
    self.logger.debug(self.bits)

    if self.bits[0] != self.START_BIT:
      self.bits = []
      return []

    if(len(self.bits) < 7):
      return []

    data_bits = []
    if bit == self.STOP_BIT:
      data_bits = self.bits[1:6]
    
    self.bits = []
    return data_bits

  def synchronise(self, input_stream):
    print("Synchronizing...")
    while True:
      chunk = input_stream.read(randrange(self.sample_rate) + int(self.sample_rate / 2))
      decoded = np.frombuffer(chunk, dtype=np.int16)
      bins = np.fft.fft(decoded)
      freqs = np.fft.fftfreq(len(bins))
      freq_amplitudes = {}
      for idx, bin in enumerate(bins):
        freq = int(abs(freqs[idx] * self.sample_rate))
        amp = np.abs(bin)
        if freq_amplitudes.get(freq):
          freq_amplitudes[freq] = freq_amplitudes.get(freq) + amp
        else:
          freq_amplitudes[freq] = amp

      sorted_by_amp = {
        k: v for k, v in sorted(freq_amplitudes.items(), key=lambda item: item[1], reverse=True)
      }

      max_amp = list(sorted_by_amp.values())[0]
      dominant_freq = list(sorted_by_amp.keys())[0]
      self.logger.debug("Dominant: " + str(dominant_freq) + "Hz @ " + str(max_amp))

      for freq in sorted_by_amp.keys():
        amp = sorted_by_amp[freq]

        if abs(dominant_freq - freq) > (self.shift / 2):
          self.logger.debug("Inactive: " + str(freq) + "Hz @ " + str(amp) + " = " + str(int(amp/max_amp*100)) + "%")
          if amp/max_amp < 0.2:
            return
          else:
            break