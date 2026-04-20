from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def clean_person_name(label: str) -> str:
    # "Boris Weirauch - ..." or "Boris Weirauch (Baden-Württemberg 2021 - 2026)"
    if " - " in label:
        label = label.split(" - ", 1)[0]
    if " (" in label:
        label = label.split(" (", 1)[0]
    return label.strip()


def clean_party_label(label: str) -> str:
    # "CDU/CSU (Bundestag 2017 - 2021)" -> "CDU/CSU"
    if " (" in label:
        label = label.split(" (", 1)[0]
    return label.strip()


def load_poll_index(polls_path: Path) -> Dict[int, Dict[str, str]]:
    index: Dict[int, Dict[str, str]] = {}
    for poll in read_jsonl(polls_path):
        poll_id = poll.get("id")
        if poll_id is None:
            continue
        legislature = (poll.get("field_legislature") or {}).get("label") or ""
        poll_date = poll.get("field_poll_date") or ""
        index[int(poll_id)] = {
            "legislature": legislature,
            "date": poll_date,
        }
    return index


def build_graph(
    votes_path: Path,
    polls_path: Path,
    parliament_label: str,
    start_date: str,
    end_date: str,
    min_co_votes: int = 0,
    exclude_votes: Iterable[str] | None = None,
) -> Tuple[nx.Graph, Dict[str, Dict[str, str]]]:
    exclude_set = {v.strip().lower() for v in (exclude_votes or []) if v.strip()}

    person_meta: Dict[str, Dict[str, str]] = {}
    poll_vote_map: Dict[int, Dict[str, str]] = defaultdict(dict)
    poll_index = load_poll_index(polls_path)

    for vote in read_jsonl(votes_path):
        vote_value = (vote.get("vote") or "unknown").strip().lower()
        if vote_value in exclude_set:
            continue

        poll_id = vote.get("_poll_id") or (vote.get("poll") or {}).get("id")
        if poll_id is None:
            continue
        poll_id = int(poll_id)
        poll_info = poll_index.get(poll_id)
        if not poll_info:
            continue
        if poll_info["legislature"] != parliament_label:
            continue
        poll_date = poll_info["date"]
        if poll_date:
            if poll_date < start_date or poll_date > end_date:
                continue

        mandate = vote.get("mandate") or {}
        person_id = str(mandate.get("id") or vote.get("id"))
        if not person_id:
            continue

        person_label = vote.get("label") or mandate.get("label") or person_id
        person_name = clean_person_name(person_label)
        fraction_raw = (vote.get("fraction") or {}).get("label") or "Unknown"
        fraction = clean_party_label(fraction_raw)

        if person_id not in person_meta:
            person_meta[person_id] = {
                "name": person_name,
                "party": fraction,
            }

        poll_vote_map[int(poll_id)][person_id] = vote_value

    joint_counts: Dict[Tuple[str, str], Dict[Tuple[str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for _, vote_map in poll_vote_map.items():
        voters = sorted(vote_map.keys())
        if len(voters) < 2:
            continue
        for u, v in combinations(voters, 2):
            vu = vote_map[u]
            vv = vote_map[v]
            joint_counts[(u, v)][(vu, vv)] += 1

    graph = nx.Graph()
    for pid, meta in person_meta.items():
        graph.add_node(pid, **meta)

    for (u, v), joint in joint_counts.items():
        total = sum(joint.values())
        if total == 0 or total < min_co_votes:
            continue

        a_counts: Dict[str, int] = defaultdict(int)
        b_counts: Dict[str, int] = defaultdict(int)
        agree = 0
        for (a, b), n in joint.items():
            a_counts[a] += n
            b_counts[b] += n
            if a == b:
                agree += n

        p0 = agree / total
        pe = 0.0
        for c in set(a_counts) | set(b_counts):
            pe += (a_counts[c] / total) * (b_counts[c] / total)

        if pe >= 1.0:
            # Degenerate case: no variability in one/both marginals.
            # If observed agreement is perfect, treat as kappa=1; else 0.
            kappa = 1.0 if p0 >= 1.0 else 0.0
        else:
            kappa = (p0 - pe) / (1.0 - pe)

        graph.add_edge(u, v, weight=kappa)

    return graph, person_meta


def count_votes(
    votes_path: Path,
    polls_path: Path,
    parliament_label: str,
    start_date: str,
    end_date: str,
    exclude_votes: Iterable[str] | None = None,
) -> Dict[str, int]:
    exclude_set = {v.strip().lower() for v in (exclude_votes or []) if v.strip()}
    poll_index = load_poll_index(polls_path)
    counts: Dict[str, int] = defaultdict(int)

    for vote in read_jsonl(votes_path):
        vote_value = (vote.get("vote") or "unknown").strip().lower()
        if vote_value in exclude_set:
            continue

        poll_id = vote.get("_poll_id") or (vote.get("poll") or {}).get("id")
        if poll_id is None:
            continue
        poll_id = int(poll_id)
        poll_info = poll_index.get(poll_id)
        if not poll_info:
            continue
        if poll_info["legislature"] != parliament_label:
            continue
        poll_date = poll_info["date"]
        if poll_date:
            if poll_date < start_date or poll_date > end_date:
                continue

        counts[vote_value] += 1

    return dict(counts)


def count_roll_calls(
    votes_path: Path,
    polls_path: Path,
    parliament_label: str,
    start_date: str,
    end_date: str,
    exclude_votes: Iterable[str] | None = None,
) -> int:
    exclude_set = {v.strip().lower() for v in (exclude_votes or []) if v.strip()}
    poll_index = load_poll_index(polls_path)
    polls_seen = set()

    for vote in read_jsonl(votes_path):
        vote_value = (vote.get("vote") or "unknown").strip().lower()
        if vote_value in exclude_set:
            continue

        poll_id = vote.get("_poll_id") or (vote.get("poll") or {}).get("id")
        if poll_id is None:
            continue
        poll_id = int(poll_id)
        poll_info = poll_index.get(poll_id)
        if not poll_info:
            continue
        if poll_info["legislature"] != parliament_label:
            continue
        poll_date = poll_info["date"]
        if poll_date:
            if poll_date < start_date or poll_date > end_date:
                continue

        polls_seen.add(poll_id)

    return len(polls_seen)


def write_outputs(
    out_dir: Path,
    graph: nx.Graph,
    person_meta: Dict[str, Dict[str, str]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = out_dir / "nodes.csv"
    edges_path = out_dir / "edges.csv"
    gexf_path = out_dir / "graph.gexf"

    with nodes_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "name", "party"])
        for pid, meta in sorted(person_meta.items(), key=lambda x: x[1]["name"]):
            writer.writerow([pid, meta["name"], meta["party"]])

    with edges_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight"])
        for u, v, data in graph.edges(data=True):
            writer.writerow([u, v, data.get("weight", 1)])

    nx.write_gexf(graph, gexf_path)


def load_graph_from_csv(nodes_path: Path, edges_path: Path) -> nx.Graph:
    graph = nx.Graph()
    with nodes_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_party = row.get("party") or "Unknown"
            graph.add_node(
                row["person_id"],
                name=row.get("name") or row["person_id"],
                party=clean_party_label(raw_party),
            )

    with edges_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            weight = float(row.get("weight") or 0.0)
            graph.add_edge(row["source"], row["target"], weight=weight)

    return graph
