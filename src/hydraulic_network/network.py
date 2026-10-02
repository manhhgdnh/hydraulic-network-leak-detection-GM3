from pathlib import Path
import numpy as np


def load_network(nodes_path, edges_path):
    """Load node coordinates and edge data from tab-separated files."""
    nodes = np.genfromtxt(Path(nodes_path), delimiter="\t")
    edges = np.genfromtxt(Path(edges_path), delimiter="\t")
    return nodes, edges


def build_pressure_system(nodes, edges, inlet, inlet_pressure, outlet, outlet_pressure):
    """Build the graph-Laplacian pressure system with Dirichlet boundary conditions."""
    n = len(nodes)
    A = np.zeros((n, n))
    b = np.zeros(n)

    for edge in edges:
        i = int(edge[0])
        j = int(edge[1])
        conductance = float(edge[2])

        A[i, i] += conductance
        A[j, j] += conductance
        A[i, j] -= conductance
        A[j, i] -= conductance

    A_bc = A.copy()
    b_bc = b.copy()

    for k in range(n):
        A_bc[inlet, k] = 0.0
        A_bc[outlet, k] = 0.0

        b_bc[k] -= A_bc[k, inlet] * inlet_pressure
        b_bc[k] -= A_bc[k, outlet] * outlet_pressure

        A_bc[k, inlet] = 0.0
        A_bc[k, outlet] = 0.0

    A_bc[inlet, inlet] = 1.0
    A_bc[outlet, outlet] = 1.0
    b_bc[inlet] = inlet_pressure
    b_bc[outlet] = outlet_pressure

    return A_bc, b_bc


def impose_dirichlet(A, b, index, value):
    """Impose p[index]=value while preserving the linear system structure."""
    A2 = A.copy()
    b2 = b.copy()

    # Adjust the right-hand side before removing the column.
    b2 -= A2[:, index] * value
    A2[index, :] = 0.0
    A2[:, index] = 0.0
    A2[index, index] = 1.0
    b2[index] = float(value)
    return A2, b2
