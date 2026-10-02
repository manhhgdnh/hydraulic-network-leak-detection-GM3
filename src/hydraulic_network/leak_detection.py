import numpy as np

from .network import impose_dirichlet
from .solvers import solve_cholesky


def candidate_nodes_near_sensors(nodes, sensor_nodes, radius=1.0, exclude=None):
    """Return nodes within a given Euclidean radius of at least one sensor."""
    exclude = set() if exclude is None else set(exclude)
    candidates = []

    for i, (x_i, y_i) in enumerate(nodes):
        if i in exclude:
            continue
        for sensor in sensor_nodes:
            x_s, y_s = nodes[sensor]
            if np.hypot(x_i - x_s, y_i - y_s) <= radius:
                candidates.append(i)
                break

    return list(dict.fromkeys(candidates))


def rank_leak_candidates(
    A_base,
    b_base,
    sensor_nodes,
    measured_pressures,
    candidates,
    leak_pressure=0.0,
):
    """Rank candidate leak nodes by RMSE against measured sensor pressures."""
    sensor_nodes = [int(node) for node in sensor_nodes]
    measured_pressures = np.asarray(measured_pressures, dtype=float)

    if len(sensor_nodes) == 0 or len(sensor_nodes) != len(measured_pressures):
        raise ValueError("Sensor indices and measured pressures must have equal non-zero length.")

    scores = []
    for node in candidates:
        A_candidate, b_candidate = impose_dirichlet(
            A_base, b_base, int(node), leak_pressure
        )
        try:
            pressure = solve_cholesky(A_candidate, b_candidate)
        except ValueError:
            continue

        prediction = pressure[sensor_nodes]
        rmse = np.linalg.norm(prediction - measured_pressures) / np.sqrt(len(sensor_nodes))
        scores.append((float(rmse), int(node), pressure))

    scores.sort(key=lambda item: item[0])
    return scores
