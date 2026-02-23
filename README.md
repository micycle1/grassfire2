# grassfire2

`grassfire2` computes straight skeletons using a kinetic-triangulation approach.
It is a modern rearchitecture of [grassfire](https://github.com/bmmeijers/grassfire), focused on maintainability, extensibility and performance.

## Getting started

```python
from grassfire2 import compute_segments

rings = [[
    [0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 10.0], [10.0, 20.0], [0.0, 20.0]
]]

segments = compute_segments(rings)
print(segments)
```

### Sync uv (dev)
`uv sync --group dev --group test`

### Run Tests
`uv run pytest tests/ -v --color=yes`

### Run Benchmark
`uv run python -O benchmarks/benchmark.py`
