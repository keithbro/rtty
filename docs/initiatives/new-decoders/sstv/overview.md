# SSTV (Slow-Scan TV) Decoder

Status: **open**

## Description
Decode SSTV images transmitted over radio. Frequency maps to pixel brightness, and scan lines build up an image row by row. A natural next step from audio-to-text into audio-to-image.

## Notes
- Multiple SSTV modes exist (Martin, Scottie, etc.) with different timing and resolution.
- A VIS (Vertical Interval Signaling) code at the start of each transmission identifies the mode.
- Output would be an image file rather than printed text.
