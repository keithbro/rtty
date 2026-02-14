import argparse
import logging
import rtty


def main():
  parser = argparse.ArgumentParser(description='Decode RTTY from a WAV file')
  parser.add_argument('wav_file', help='Path to WAV file')
  parser.add_argument('--baud', type=int, default=50)
  parser.add_argument('--shift', type=int, default=450, help='Frequency shift in Hz')
  parser.add_argument('--stop-bits', type=float, default=1.5)
  parser.add_argument('--no-reverse-bits', action='store_true')
  parser.add_argument('--no-invert', action='store_true')
  parser.add_argument('--skip-sync', action='store_true', help='Skip synchronization phase')
  parser.add_argument('--sync-timeout', type=float, default=30,
                      help='Sync timeout in seconds (0 for no timeout, default: 30)')
  args = parser.parse_args()

  logger = logging.getLogger('rtty')
  logger.setLevel(logging.DEBUG)
  ch = logging.FileHandler('debug.log')
  ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
  logger.addHandler(ch)

  stream = rtty.WavStream(args.wav_file)
  decoder = rtty.Decoder(
    stream.sample_rate, args.baud, args.shift, args.stop_bits,
    not args.no_invert, logger,
  )
  ita2 = rtty.Ita2(not args.no_reverse_bits, logger)

  if not args.skip_sync:
    timeout = args.sync_timeout if args.sync_timeout > 0 else None
    decoder.synchronise(stream, timeout=timeout)

  while True:
    chunk_size = decoder.chunk_size()
    chunk = stream.read(chunk_size)

    if len(chunk) < chunk_size * 2:  # 2 bytes per 16-bit sample
      break

    bit = decoder.decode(chunk)

    if bit is None:
      continue

    data_bits = decoder.process(bit)

    if len(data_bits) == 0:
      continue

    symbol = ita2.decode(data_bits)

    if symbol:
      print(symbol, end="", flush=True)

  print()
  stream.close()


if __name__ == '__main__':
  main()
