# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RTTY (Radio Teletype) signal decoder — a Python learning experiment that decodes FSK-modulated audio into readable text using ITA2 character encoding. Tested against real signals at 4582.97KHz and sample signals from sigidwiki.com.

## Running

```bash
python3 main.py
```

Dependencies are in `requirements.txt` (`numpy`, `pyaudio`, `pytest`). Install with `pip3 install -r requirements.txt`.

Audio device indices in `main.py` are hardcoded (input=4, output=1) and will need adjusting per machine.

To decode a WAV file offline:

```bash
python3 decode_wav.py --skip-sync tests/fixtures/rtty_450hz_50bd.wav
```

Flags: `--skip-sync`, `--no-invert`, `--no-reverse-bits`, `--baud`, `--shift`, `--stop-bits`, `--waterfall`.

To decode a WAV file with a spectrogram visualization (requires `matplotlib`):

```bash
python3 decode_wav.py --skip-sync --waterfall tests/fixtures/rtty_450hz_50bd.wav
```

## Architecture

The signal pipeline flows: **Audio Input → FFT decode → bit processing → ITA2 character lookup → printed text**.

Four components:

- **`main.py`** — Live entry point. Configures PyAudio streams (mono, 16-bit, 8kHz), runs synchronization, then loops reading audio chunks through the decoder. Also loops audio back to an output device. Hardcoded params: sample_rate=8000, baud=50, shift=450Hz, stop_bits=1.5.

- **`decode_wav.py`** — Offline entry point. Decodes RTTY from a WAV file using the same `Decoder` → `Ita2` pipeline. Takes CLI args for RTTY params.

- **`rtty/decoder.py`** (`Decoder`) — Core signal processing. `synchronise()` calibrates by finding the dominant and inactive frequencies. `decode(chunk)` runs NumPy FFT on each audio chunk to determine if the current bit is mark (1) or space (0), with confidence scoring. `process(bit)` accumulates bits and extracts 5-bit data frames (1 start bit + 5 data bits + 1 stop bit).

- **`rtty/ita2.py`** (`Ita2`) — Translates 5-bit patterns to characters. Maintains letter/figure shift state. Two lookup tables (LETTERS, FIGURES) switched by special shift characters.

- **`rtty/wav_stream.py`** (`WavStream`) — Thin adapter around Python's `wave` module. Exposes `.read(num_samples)` matching the PyAudio stream interface so `Decoder` works identically with both live and file sources.

## Tests

Run tests with:

```bash
python3 -m pytest tests/
```

- `tests/test_ita2.py` — Unit tests for ITA2 character decoding and shift modes
- `tests/test_decoder.py` — Unit tests for bit framing and chunk sizing
- `tests/test_wav_decode.py` — Integration tests decoding `tests/fixtures/rtty_450hz_50bd.wav` end-to-end (expects "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 0123456789")

## Project Planning

Work is tracked in `docs/initiatives/` using a three-level hierarchy:

- **Initiatives** — Top-level folders representing broad goals (e.g. `engineering-health/`, `new-decoders/`).
- **Features** — Subfolders under an initiative for multi-task efforts (e.g. `new-decoders/cw-morse/`). Contains an `overview.md` describing the feature's scope.
- **Tasks** — Individual `.md` files describing a single piece of work. Can live directly under an initiative (for standalone tasks) or under a feature folder.

Task files follow this format:

```markdown
# Title

Status: **open** | **done**
PR: (link when done)

## Description
What and why.

## Notes
Implementation hints, references.
```

When raising a PR that completes a task, include the task file update in the same PR: set the status to **done** and add the PR link. This way the task is marked done as part of the merge — no follow-up needed.

## Code Style

- Always use `python3` (not `python`) in commands and instructions.
- Every new function should have a one-line comment explaining what it does.

## Workflow Preferences

- When asked to "pick up" a task: choose which task yourself if the user hasn't specified one, and always commit and open a PR when finished.
