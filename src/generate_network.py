#!/usr/bin/env python3
"""Generate a random 7x7 hydraulic network compatible with the project data format."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "generated"
OUTPUT.mkdir(parents=True, exist_ok=True)

nodes = []
edges = []

n = 7
spacing = 0.5
threshold_base = 1.6

for j in range(n):
    y = j * spacing
    for i in range(n):
        x = i * spacing
        nodes.append([x, y])

        if i < n - 1 and j < n - 1:
            idx = j * n + i
            right = idx + 1
            down = (j + 1) * n + i
            diagonal = down + 1

            if np.random.random() > 0.5:
                edges.append([idx, right, 1.0, threshold_base + np.random.random()])
            else:
                edges.append([idx, down, 1.0, threshold_base + np.random.random()])

            if np.random.random() > 0.5:
                edges.append([idx, diagonal, 1.0, threshold_base + np.random.random()])
            else:
                edges.append([right, down, 1.0, threshold_base + np.random.random()])

        if j == n - 1 and i < n - 1:
            idx = j * n + i
            edges.append([idx, idx + 1, 1.0, threshold_base + np.random.random()])

        if i == n - 1 and j < n - 1:
            idx = j * n + i
            edges.append([idx, (j + 1) * n + i, 1.0, threshold_base + np.random.random()])

np.savetxt(OUTPUT / "nodes.tsv", nodes, delimiter="\t")
np.savetxt(OUTPUT / "edges.tsv", edges, delimiter="\t")
print(f"Generated network written to {OUTPUT}")
