# grassfire2

`grassfire2` computes straight skeletons using a kinetic-triangulation approach.
It is a modern rearchitecture of [grassfire](https://github.com/bmmeijers/grassfire), focused on maintainability, extensibility and performance.

## Getting started

```python
from tri.delaunay.helpers import ToPointsAndSegments
from grassfire2 import compute_segments

conv = ToPointsAndSegments()
conv.add_polygon(
    [[
        [0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 10.0], [10.0, 20.0], [0.0, 20.0]
    ]]
)

segments = compute_segments(conv)
print(segments)
```

### Sync uv (dev)
`uv sync --all-extras --group dev --group test`

### Run Tests
`uv run pytest tests/ -v --color=yes`

### Run Benchmark
`uv run python benchmarks/benchmark_micycle.py`