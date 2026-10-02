import numpy as np
import matplotlib.pyplot as plt


def plot_network(nodes, edges, ax=None):
    """Plot the hydraulic network topology."""
    if ax is None:
        _, ax = plt.subplots()

    for edge in edges:
        p1 = nodes[int(edge[0])]
        p2 = nodes[int(edge[1])]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-")

    for idx, (x, y) in enumerate(nodes):
        ax.scatter([x], [y], s=160, c="black", zorder=5)
        ax.text(x, y, str(idx), ha="center", va="center", fontsize=8, color="white", zorder=6)

    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_pressure_network(nodes, edges, pressures, inlets=None, outlets=None, ax=None):
    """Plot flow directions and node pressures on the network."""
    inlets = [] if inlets is None else list(inlets)
    outlets = [] if outlets is None else list(outlets)

    if ax is None:
        _, ax = plt.subplots()

    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        p1, p2 = nodes[i], nodes[j]
        threshold = float(edge[3])
        delta = pressures[j] - pressures[i]
        color = "red" if abs(delta) > threshold else "blue"

        if delta < 0:
            start, end = p1, p2
        elif delta > 0:
            start, end = p2, p1
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)
            start = end = None

        if start is not None:
            ax.arrow(
                start[0],
                start[1],
                end[0] - start[0],
                end[1] - start[1],
                color=color,
                width=0.01,
                head_width=0.1,
                length_includes_head=True,
            )

    for idx, (x, y) in enumerate(nodes):
        marker_color = "red" if idx in inlets else "green" if idx in outlets else "black"
        ax.scatter([x], [y], s=180, c=marker_color, zorder=5)
        ax.text(x, y, str(idx), ha="center", va="center", fontsize=8, color="white", zorder=6)
        ax.text(x, y + 0.18, f"{pressures[idx]:.2f}", ha="center", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_candidate_scores(scores, top_n=10, ax=None):
    """Plot the RMSE of the best leak candidates."""
    if ax is None:
        _, ax = plt.subplots()

    selected = scores[:top_n]
    nodes = [node for rmse, node, _ in selected]
    rmses = [rmse for rmse, node, _ in selected]
    positions = np.arange(len(selected))

    ax.bar(positions, rmses)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(node) for node in nodes])
    ax.set_xlabel("Candidate leak node")
    ax.set_ylabel("RMSE")
    ax.set_title("Top leak candidates")
    ax.grid(axis="y", alpha=0.25)
    return ax
