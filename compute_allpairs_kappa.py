"""
Compute all-pairs Cohen's kappa from raw.json vote data.
Writes edges_allpairs.csv alongside existing edges.csv — never overwrites it.
"""
import json
import csv
import sys
from itertools import combinations
from pathlib import Path
from collections import defaultdict

MIN_CO_VOTES = int(sys.argv[2]) if len(sys.argv) > 2 else 5
PERIODS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "bundestag_2005_2009", "bundestag_2009_2013", "bundestag_2013_2017",
    "bundestag_2017_2021", "bundestag_2021_2025", "bundestag_2025_2029",
]

OUTPUT = Path("output")

VOTE_VALUES = {"yes", "no", "abstain", "no_show"}

def compute_kappa(votes_a, votes_b):
    """Cohen's kappa on shared polls."""
    shared = set(votes_a) & set(votes_b)
    if len(shared) < 1:
        return None, 0
    n = len(shared)
    agree = sum(1 for p in shared if votes_a[p] == votes_b[p])
    p0 = agree / n

    # expected agreement
    counts = defaultdict(int)
    for p in shared:
        counts[votes_a[p]] += 1
        counts[votes_b[p]] += 1
    total = 2 * n
    pe = sum((c / total) ** 2 for c in counts.values())

    if pe >= 1.0:
        return (1.0 if p0 >= 1.0 else 0.0), n
    return (p0 - pe) / (1.0 - pe), n


for period in PERIODS:
    data_path = OUTPUT / period / "raw.json"
    out_path  = OUTPUT / period / "edges_allpairs.csv"

    if not data_path.exists():
        print(f"  {period}: raw.json not found, skipping")
        continue

    print(f"{period}: loading…", flush=True)
    with open(data_path) as f:
        raw = json.load(f)

    # Build person → {poll_id: vote} map
    person_votes: dict[str, dict[int, str]] = defaultdict(dict)
    for vote in raw["votes"]:
        person = vote.get("mandate", {}) or {}
        pid    = str(person.get("id") or "")
        if not pid:
            continue
        poll_id = vote.get("poll", {}).get("id")
        if poll_id is None:
            continue
        v = (vote.get("vote") or "").strip().lower()
        if v not in VOTE_VALUES:
            continue
        person_votes[pid][int(poll_id)] = v

    persons = sorted(person_votes)
    print(f"  {len(persons)} MPs, {len(raw['polls'])} polls", flush=True)

    written = 0
    skipped = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "weight"])
        for a, b in combinations(persons, 2):
            kappa, n_shared = compute_kappa(person_votes[a], person_votes[b])
            if kappa is None or n_shared < MIN_CO_VOTES:
                skipped += 1
                continue
            w.writerow([a, b, f"{kappa:.6f}"])
            written += 1

    print(f"  → {written} pairs written, {skipped} skipped (min_co_votes={MIN_CO_VOTES})")
    print(f"  Saved: {out_path}")
