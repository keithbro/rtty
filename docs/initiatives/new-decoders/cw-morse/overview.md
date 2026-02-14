# CW (Morse Code) Decoder

## Description
Decode CW (Morse code) signals from audio. Morse uses on/off keying of a single tone rather than FSK with two frequencies, introducing variable-length character encoding and timing-based detection of dots, dashes, and gaps.

## Notes
- Detect presence/absence of a single tone rather than comparing two frequencies.
- Timing thresholds distinguish dots from dashes and character gaps from word gaps.
- A good reference for standard Morse timing: dot = 1 unit, dash = 3 units, intra-character gap = 1 unit, inter-character gap = 3 units, word gap = 7 units.
