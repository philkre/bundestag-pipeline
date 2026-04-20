#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from pipeline.cli import parse_args
from pipeline.ingest import (
    build_graph,
    count_roll_calls,
    count_votes,
    load_graph_from_csv,
    write_outputs,
)
from pipeline.network import basic_metrics, cross_party_edge_counts
from pipeline.plot import load_party_colors, plot_graph, plot_weight_histogram


def main() -> None:
    args = parse_args()
    votes_path = Path(args.votes)
    polls_path = Path(args.polls)
    out_dir = Path(args.out_dir)

    nodes_path = out_dir / "nodes.csv"
    edges_path = out_dir / "edges.csv"

    used_csv = False
    if args.force or not (nodes_path.exists() and edges_path.exists()):
        graph, person_meta = build_graph(
            votes_path,
            polls_path,
            parliament_label=args.parliament,
            start_date=args.start_date,
            end_date=args.end_date,
            min_co_votes=args.min_co_votes,
            exclude_votes=args.exclude_votes,
        )
        write_outputs(out_dir, graph, person_meta)
    else:
        graph = load_graph_from_csv(nodes_path, edges_path)
        used_csv = True

    plot_path = out_dir / args.plot
    party_colors = load_party_colors(Path(args.party_colors))

    plot_graph(
        graph,
        plot_path,
        min_weight=args.min_weight,
        only_cross_party=args.only_cross_party,
        with_labels=args.labels,
        party_colors=party_colors,
        metric_name=args.metric_name,
        layout=args.layout,
    )

    if args.histogram:
        hist_path = out_dir / "edge_weight_hist.png"
        plot_weight_histogram(
            graph, hist_path, bins=args.hist_bins, metric_name=args.metric_name
        )
        print(f"Wrote: {hist_path}")

    metrics = basic_metrics(graph)
    print(
        f"Nodes: {int(metrics['nodes'])}  Edges: {int(metrics['edges'])}  "
        f"Density: {metrics['density']:.4f}  Avg degree: {metrics['avg_degree']:.2f}  "
        f"Components: {int(metrics['components'])}"
    )
    vote_counts = count_votes(
        votes_path,
        polls_path,
        parliament_label=args.parliament,
        start_date=args.start_date,
        end_date=args.end_date,
        exclude_votes=args.exclude_votes,
    )
    roll_calls = count_roll_calls(
        votes_path,
        polls_path,
        parliament_label=args.parliament,
        start_date=args.start_date,
        end_date=args.end_date,
        exclude_votes=args.exclude_votes,
    )
    if vote_counts:
        total_votes = sum(vote_counts.values())
        parts = [f"{k}={v}" for k, v in sorted(vote_counts.items())]
        print(f"Votes: total={total_votes}  " + "  ".join(parts))
    print(f"Roll calls: {roll_calls}")
    if args.only_cross_party:
        counts = cross_party_edge_counts(graph)
        print(
            f"Cross-party edges: {counts['cross_party_edges']} / {counts['total_edges']}"
        )
    if used_csv:
        print("Note: loaded graph from CSV (use --force to rebuild from JSONL).")
    if args.force or not (nodes_path.exists() and edges_path.exists()):
        print(f"Wrote: {out_dir / 'nodes.csv'}")
        print(f"Wrote: {out_dir / 'edges.csv'}")
        print(f"Wrote: {out_dir / 'graph.gexf'}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
