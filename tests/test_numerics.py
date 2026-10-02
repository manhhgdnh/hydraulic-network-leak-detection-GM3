import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydraulic_network.network import build_pressure_system, load_network
from hydraulic_network.solvers import solve_cholesky, solve_cramer
from hydraulic_network.leak_detection import rank_leak_candidates


class NumericalTests(unittest.TestCase):
    def test_cholesky_matches_numpy(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        expected = np.linalg.solve(A, b)
        actual = solve_cholesky(A, b)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_cramer_matches_numpy(self):
        A = np.array([[3.0, -1.0], [2.0, 4.0]])
        b = np.array([7.0, 10.0])
        expected = np.linalg.solve(A, b)
        actual = solve_cramer(A, b)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_project_leak_ranking(self):
        nodes, edges = load_network(
            ROOT / "data" / "large" / "nodes.tsv",
            ROOT / "data" / "large" / "edges.tsv",
        )
        inlet, outlet = 7, len(nodes) - 1
        A, b = build_pressure_system(nodes, edges, inlet, 10.0, outlet, 0.0)
        candidates = [i for i in range(len(nodes)) if i not in {inlet, outlet}]
        scores = rank_leak_candidates(
            A,
            b,
            [6, 22, 46],
            [4.96, 6.98, 3.51],
            candidates,
            leak_pressure=0.0,
        )
        self.assertTrue(scores)
        self.assertEqual(scores[0][1], 34)


if __name__ == "__main__":
    unittest.main()
