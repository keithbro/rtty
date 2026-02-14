# Add synchronisation timeout

Status: **open**

## Description
Handle the case where `synchronise()` loops forever when no signal is present. Add a configurable timeout so the decoder gives up gracefully instead of hanging indefinitely.

## Notes
- Currently `synchronise()` blocks until it finds two dominant frequencies, which never happens with silence or noise-only input.
- A timeout with a clear error message would improve the user experience for live decoding.
