# PSK31 Decoder

Status: **open**

## Description
Decode PSK31 signals, which use phase-shift keying instead of frequency-shift keying. This teaches a fundamentally different modulation scheme and introduces Varicode, a variable-length character encoding.

## Notes
- PSK31 modulates data by shifting the phase of a ~1 kHz carrier by 180 degrees.
- Varicode uses variable-length bit patterns (shorter for common characters like 'e', longer for rare ones).
- Costas loop or similar technique needed for carrier phase recovery.
