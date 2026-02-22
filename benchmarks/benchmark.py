import csv
from pathlib import Path
from pyinstrument import Profiler
from grassfire2 import compute_skeleton
from tri.delaunay.helpers import ToPointsAndSegments

def read_csv_polygon(path):
    rings = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            coords = [float(x) for x in row if x.strip()]
            points = []
            for i in range(0, len(coords), 2):
                points.append((coords[i], coords[i+1]))
            if points and points[0] != points[-1]:
                points.append(points[0])
            if points:
                rings.append(points)
    return rings

def benchmark_large():
    base_path = Path(__file__).parent.parent
    csv_path = base_path / "tests" / "integration" / "csv" / "micycle-1.csv"
    
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    rings = read_csv_polygon(csv_path)
    print(f"Loaded {len(rings)} rings from {csv_path.name}")

    profiler = Profiler()
    profiler.start()
    
    N = 10
    print(f"Starting computation ({N} iterations)...")
    for _ in range(N):
        conv = ToPointsAndSegments()
        conv.add_polygon(rings)
        sk = compute_skeleton(conv)
    print("Computation finished.")
    
    profiler.stop()
    
    profiler.print()
    
    # Optional: save to HTML
    with open("micycle_profile.html", "w") as f:
        f.write(profiler.output_html())

if __name__ == "__main__":
    benchmark_large()
