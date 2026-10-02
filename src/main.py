#!/usr/bin/env python3
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow direct execution with: python3 src/main.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydraulic_network.leak_detection import (  # noqa: E402
    candidate_nodes_near_sensors,
    rank_leak_candidates,
)
from hydraulic_network.network import build_pressure_system, load_network  # noqa: E402
from hydraulic_network.solvers import solve_cholesky, solve_cramer  # noqa: E402
from hydraulic_network.visualization import (  # noqa: E402
    plot_candidate_scores,
    plot_network,
    plot_pressure_network,
)

DATA = ROOT / "data"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def save_current_figure(filename):
    path = FIGURES / filename
    plt.gcf().savefig(path, dpi=180, bbox_inches="tight")
    print(f"Figure saved to: {path.relative_to(ROOT)}")



def choose_network(choice):
    if choice == "1":
        return load_network(DATA / "simple" / "nodes.tsv", DATA / "simple" / "edges.tsv")
    return load_network(DATA / "large" / "nodes.tsv", DATA / "large" / "edges.tsv")


def visualize_network():
    choice = input("Network: 1) simple  2) large: ").strip()
    nodes, edges = choose_network(choice)
    plot_network(nodes, edges)
    name = "network_simple.png" if choice == "1" else "network_large.png"
    save_current_figure(name)
    plt.show()


def solve_pressure_problem():
    choice = input("Network: 1) simple  2) large: ").strip()
    nodes, edges = choose_network(choice)

    method = input("Solver: 1) Cramer  2) Cholesky: ").strip()
    inlet, outlet = 0, len(nodes) - 1
    A, b = build_pressure_system(nodes, edges, inlet, 10.0, outlet, 0.0)

    if method == "1":
        if len(nodes) > 10:
            raise ValueError("Cramer's rule is intentionally limited to the small network.")
        pressure = solve_cramer(A, b)
    else:
        pressure = solve_cholesky(A, b)

    print("Pressure vector:")
    print(np.array2string(pressure, precision=4))
    plot_pressure_network(nodes, edges, pressure, inlets=[inlet], outlets=[outlet])
    network_name = "simple" if choice == "1" else "large"
    solver_name = "cramer" if method == "1" else "cholesky"
    save_current_figure(f"pressure_{network_name}_{solver_name}.png")
    plt.show()


def detect_leak():
    nodes, edges = load_network(DATA / "large" / "nodes.tsv", DATA / "large" / "edges.tsv")
    n = len(nodes)

    inlet, inlet_pressure = 7, 10.0
    outlet, outlet_pressure = n - 1, 0.0
    sensor_nodes = [6, 22, 46]
    measured_pressures = np.array([4.96, 6.98, 3.51])

    A0, b0 = build_pressure_system(
        nodes, edges, inlet, inlet_pressure, outlet, outlet_pressure
    )

    strategy = input("Candidates: 1) all nodes  2) near sensors: ").strip()
    if strategy == "2":
        try:
            radius = float(input("Radius R [default 1.0]: ").strip() or "1.0")
        except ValueError:
            radius = 1.0
        candidates = candidate_nodes_near_sensors(
            nodes, sensor_nodes, radius=radius, exclude={inlet, outlet}
        )
        if not candidates:
            candidates = [i for i in range(n) if i not in {inlet, outlet}]
    else:
        candidates = [i for i in range(n) if i not in {inlet, outlet}]

    scores = rank_leak_candidates(
        A0,
        b0,
        sensor_nodes,
        measured_pressures,
        candidates,
        leak_pressure=outlet_pressure,
    )

    if not scores:
        print("No valid leak candidate could be evaluated.")
        return

    print("\nTop leak candidates (smaller RMSE is better):")
    for rank, (rmse, node, _) in enumerate(scores[:10], start=1):
        print(f"{rank:2d}. node {node:2d} | RMSE = {rmse:.4e}")

    best_rmse, best_node, best_pressure = scores[0]
    print(f"\nMost plausible leak: node {best_node} (RMSE = {best_rmse:.4e})")

    plot_pressure_network(
        nodes,
        edges,
        best_pressure,
        inlets=[inlet],
        outlets=[outlet, best_node],
    )
    save_current_figure(f"leak_detection_node_{best_node}.png")
    plt.show()

    plot_candidate_scores(scores, top_n=10)
    save_current_figure("leak_candidate_ranking.png")
    plt.show()


def main():
    print("Hydraulic Network Leak Detection")
    print("1. Visualize a network")
    print("2. Solve the pressure system")
    print("3. Detect the most plausible leak")
    choice = input("Choice: ").strip()

    if choice == "1":
        visualize_network()
    elif choice == "2":
        solve_pressure_problem()
    elif choice == "3":
        detect_leak()
    else:
        print("Unknown choice.")


if __name__ == "__main__":
    main()
