# Waterfall/Spectrogram Visualization

## Description
Use matplotlib to render a live spectrogram while decoding. Seeing the two FSK tones visually helps with debugging and is a great way to understand what the decoder is actually doing.

## Notes
- matplotlib's animation API can update a spectrogram in near real-time.
- Could reuse the FFT data already computed by the decoder to avoid redundant work.
- Consider making this an optional flag (e.g. `--waterfall`) so it doesn't add a hard dependency on matplotlib for normal use.
