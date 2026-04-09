from __future__ import annotations

from typing import Dict

import networkx as nx


def basic_metrics(graph: nx.Graph) -> Dict[str, float]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if n == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0.0,
            "avg_degree": 0.0,
            "components": 0,
        }

    avg_degree = (2 * m) / n
    density = nx.density(graph)
    components = nx.number_connected_components(graph)

    return {
        "nodes": float(n),
        "edges": float(m),
        "density": float(density),
        "avg_degree": float(avg_degree),
        "components": float(components),
    }


def cross_party_edge_counts(graph: nx.Graph) -> Dict[str, int]:
    total = graph.number_of_edges()
    cross = 0
    for u, v in graph.edges():
        if graph.nodes[u].get("party") != graph.nodes[v].get("party"):
            cross += 1
    return {"total_edges": total, "cross_party_edges": cross}
