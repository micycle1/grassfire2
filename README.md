# grassfire2

`grassfire2` computes straight skeletons using a kinetic-triangulation approach.
It is a modern rearchitecture of [grassfire](https://github.com/bmmeijers/grassfire), focused on a smaller core API.

## Getting started

```python
from tri.delaunay.helpers import ToPointsAndSegments
from grassfire2 import compute_segments

conv = ToPointsAndSegments()
conv.add_polygon(
    [[
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 3.0),
        (0.0, 3.0),
        (0.0, 0.0),  # ring must be closed
    ]]
)

segments = compute_segments(conv)
print(segments[:3])  # sample of straight-skeleton segments
```
