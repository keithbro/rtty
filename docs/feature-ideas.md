# Feature Ideas

Future directions for the project, roughly ordered by complexity.

## Waterfall/spectrogram visualization

Use matplotlib to render a live spectrogram while decoding. Seeing the two FSK tones visually helps with debugging and is a great way to understand what the decoder is actually doing.

## Decode CW (Morse code)

Morse uses on/off keying of a single tone rather than FSK with two frequencies. More interesting character framing logic — variable-length encoding and timing-based detection of dots, dashes, and gaps.

## SSTV (Slow-Scan TV) decoder

Decode images transmitted over radio. Frequency maps to pixel brightness, scan lines build up an image row by row. A natural next step from audio-to-text into audio-to-image.

## PSK31 decoder

Phase-shift keying instead of frequency-shift keying. Teaches a fundamentally different modulation scheme and introduces varicode (variable-length character encoding).

## Code quality improvements

- Extract duplicated FFT-to-frequency-amplitude logic (shared between `decode()` and `synchronise()`)
- Rename `bin` variable to avoid shadowing the Python builtin
- Replace `list.pop(0)` with `collections.deque(maxlen=50)` for O(1) pops
- Handle the case where `synchronise()` loops forever with no signal (timeout)
