from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def normalize_party_key(label: str) -> str:
    folded = label.casefold().strip()
    folded = (
        folded.replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
    )
    return " ".join(folded.replace("/", " ").split())


def resolve_party_color(party: str, party_colors: Dict[str, str] | None) -> str | None:
    if not party_colors:
        return None

    # Direct match
    if party in party_colors:
        return party_colors[party]

    # Normalized match
    norm_map = {normalize_party_key(k): v for k, v in party_colors.items()}
    norm_party = normalize_party_key(party)
    if norm_party in norm_map:
        return norm_map[norm_party]

    # Aliases
    if "cdu/csu" in party.lower() and "CDU" in party_colors:
        return party_colors["CDU"]
    if "bundnis 90" in norm_party and "Bündnis 90/Die Grünen" in party_colors:
        return party_colors["Bündnis 90/Die Grünen"]
    if "grun" in norm_party and "Bündnis 90/Die Grünen" in party_colors:
        return party_colors["Bündnis 90/Die Grünen"]

    return None


def load_party_colors(path: Path) -> Dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_graph(
    graph: nx.Graph,
    out_path: Path,
    min_weight: float = 0.0,
    only_cross_party: bool = False,
    with_labels: bool = False,
    party_colors: Dict[str, str] | None = None,
    metric_name: str = "Cohen's kappa",
    layout: str = "party",
) -> None:
    if graph.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes; check input data.")

    edges = [(u, v, d) for u, v, d in graph.edges(data=True)]
    if min_weight is not None:
        edges = [e for e in edges if e[2].get("weight", 0) > min_weight]

    parties = sorted(
        {data.get("party", "Unknown") for _, data in graph.nodes(data=True)}
    )
    color_map: Dict[str, str] = {}
    cmap = plt.get_cmap("tab20")
    for i, party in enumerate(parties):
        resolved = resolve_party_color(party, party_colors)
        if resolved:
            color_map[party] = resolved
        else:
            color_map[party] = cmap(i % cmap.N)

    node_colors = [
        color_map[data.get("party", "Unknown")] for _, data in graph.nodes(data=True)
    ]

    # Party-based layout: fixed centroids, jitter within each party blob.
    party_to_nodes: Dict[str, List[str]] = defaultdict(list)
    for node, data in graph.nodes(data=True):
        party_to_nodes[data.get("party", "Unknown")].append(node)

    parties_by_size = sorted(
        party_to_nodes.keys(),
        key=lambda p: (-len(party_to_nodes[p]), p),
    )
    num_parties = max(len(parties_by_size), 1)
    radius = 10.0
    centroids = {}
    for i, party in enumerate(parties_by_size):
        angle = 2 * math.pi * (i / num_parties)
        centroids[party] = (radius * math.cos(angle), radius * math.sin(angle))

    rng = __import__("random").Random(42)
    pos = {}
    for party, nodes in party_to_nodes.items():
        cx, cy = centroids[party]
        n = len(nodes)
        spread = 0.8 + 0.15 * math.sqrt(max(n, 1))
        for node in nodes:
            dx = rng.uniform(-spread, spread)
            dy = rng.uniform(-spread, spread)
            pos[node] = (cx + dx, cy + dy)

    if layout == "force_atlas2":
        try:
            from fa2 import ForceAtlas2
        except Exception:
            layout = "party"
        else:
            fa = ForceAtlas2(
                outboundAttractionDistribution=True,
                linLogMode=False,
                adjustSizes=False,
                edgeWeightInfluence=1.0,
                jitterTolerance=1.0,
                barnesHutOptimize=True,
                barnesHutTheta=1.2,
                scalingRatio=2.0,
                strongGravityMode=False,
                gravity=1.0,
                verbose=False,
            )
            pos = fa.forceatlas2_networkx_layout(graph, pos=pos, iterations=200)

    # Edge selection:
    # - intra-party: draw all, very light
    # - cross-party: keep top 10 edges per node by weight
    intra_edges = []
    cross_edges = []
    for u, v, d in edges:
        if graph.nodes[u].get("party") == graph.nodes[v].get("party"):
            if not only_cross_party:
                intra_edges.append((u, v))
        else:
            cross_edges.append((u, v, d))

    # top 10 cross-party edges per node
    top_k = 10
    by_node: Dict[str, List[tuple]] = defaultdict(list)
    for u, v, d in cross_edges:
        by_node[u].append((u, v, d))
        by_node[v].append((u, v, d))

    keep = set()
    for node, lst in by_node.items():
        lst_sorted = sorted(lst, key=lambda e: e[2].get("weight", 0), reverse=True)
        for e in lst_sorted[:top_k]:
            a, b = e[0], e[1]
            keep.add((a, b) if a <= b else (b, a))

    cross_edges = [
        (u, v)
        for (u, v, d) in cross_edges
        if (u, v) in keep or (v, u) in keep
    ]

    plt.figure(figsize=(14, 10))
    if intra_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=intra_edges,
            alpha=0.03,
            width=0.4,
            edge_color="#444444",
        )
    if cross_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=cross_edges,
            alpha=0.5,
            width=0.8,
            edge_color="#444444",
        )
    nx.draw_networkx_nodes(
        graph, pos, node_size=80, node_color=node_colors, linewidths=0
    )

    if with_labels:
        labels = {n: data.get("name", n) for n, data in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=party,
            markerfacecolor=color_map[party],
            markersize=8,
        )
        for party in parties
    ]
    plt.legend(handles=handles, title="Party", loc="best", fontsize=8, title_fontsize=9)
    plt.title(f"Voting Similarity Network ({metric_name})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_weight_histogram(
    graph: nx.Graph,
    out_path: Path,
    bins: int = 50,
    metric_name: str = "Cohen's kappa",
) -> None:
    all_weights = [data.get("weight", 1) for _, _, data in graph.edges(data=True)]
    same_party = [
        data.get("weight", 1)
        for u, v, data in graph.edges(data=True)
        if graph.nodes[u].get("party") == graph.nodes[v].get("party")
    ]
    cross_party = [
        data.get("weight", 1)
        for u, v, data in graph.edges(data=True)
        if graph.nodes[u].get("party") != graph.nodes[v].get("party")
    ]
    if not all_weights:
        raise ValueError("No edges available for histogram.")

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    ax_top = fig.add_subplot(gs[0, :])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    counts_all, bin_edges = np.histogram(all_weights, bins=bins)
    counts_all = counts_all / counts_all.max() if counts_all.max() > 0 else counts_all
    ax_top.bar(
        bin_edges[:-1],
        counts_all,
        width=np.diff(bin_edges),
        align="edge",
        color="#10243E",
        edgecolor="#ffffff",
        linewidth=0.5,
    )
    ax_top.set_title("All")
    ax_top.set_xlabel(metric_name)
    ax_top.set_ylabel("Normalized frequency")

    counts_same, bin_edges = np.histogram(same_party, bins=bins)
    counts_same = (
        counts_same / counts_same.max() if counts_same.max() > 0 else counts_same
    )
    ax_left.bar(
        bin_edges[:-1],
        counts_same,
        width=np.diff(bin_edges),
        align="edge",
        color="#1F4B6B",
        edgecolor="#ffffff",
        linewidth=0.5,
    )
    ax_left.set_title("Same-Party")
    ax_left.set_xlabel(metric_name)
    ax_left.set_ylabel("Normalized frequency")

    counts_cross, bin_edges = np.histogram(cross_party, bins=bins)
    counts_cross = (
        counts_cross / counts_cross.max() if counts_cross.max() > 0 else counts_cross
    )
    ax_right.bar(
        bin_edges[:-1],
        counts_cross,
        width=np.diff(bin_edges),
        align="edge",
        color="#3D6A91",
        edgecolor="#ffffff",
        linewidth=0.5,
    )
    ax_right.set_title("Cross-Party")
    ax_right.set_xlabel(metric_name)
    ax_right.set_ylabel("Normalized frequency")

    plt.suptitle(f"Edge Weight Distribution ({metric_name})", fontsize=16)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
