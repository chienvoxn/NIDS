"""Heuristic communication topology classification."""

import math


def classify_topology(nodes, edges, max_degree, mean_degree):

    values = [nodes, edges, max_degree, mean_degree]

    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Invalid graph statistics")

    if nodes < 3 or edges < 2:
        return "INSUFFICIENT_DATA"

    concentration = max_degree / (nodes - 1)

    if concentration >= 0.8 and max_degree >= 3 * mean_degree:
        return "CENTRALIZED"

    if concentration <= 0.5 and mean_degree >= 2:
        return "P2P"

    return "UNDETERMINED"
