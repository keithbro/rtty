# Replace `list.pop(0)` with `deque`

Status: **open**

## Description
Replace the `list.pop(0)` pattern with `collections.deque(maxlen=50)` for O(1) pops instead of O(n) list shifts.

## Notes
- `list.pop(0)` is O(n) because it shifts all remaining elements.
- `deque` with a `maxlen` automatically discards old entries, removing the need for manual pop logic.
