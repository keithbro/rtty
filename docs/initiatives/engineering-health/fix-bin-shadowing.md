# Fix `bin` variable shadowing

## Description
Rename the `bin` variable in `rtty/decoder.py` to avoid shadowing the Python builtin `bin()`. Use a more descriptive name like `freq_bin` or `fft_bin`.

## Notes
- Shadowing builtins can cause subtle bugs if the builtin is ever needed in the same scope.
- A simple find-and-replace within the decoder module.
