# Extract duplicated FFT logic

Status: **open**

## Description
Extract the duplicated FFT-to-frequency-amplitude logic that is shared between `decode()` and `synchronise()` in `rtty/decoder.py` into a shared helper method. This reduces duplication and makes the FFT logic easier to modify in one place.

## Notes
- Both methods perform an FFT on an audio chunk and find the dominant frequency bin.
- A private method like `_fft_peak(chunk)` returning the dominant frequency and amplitude would cover both call sites.
