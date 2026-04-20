"""
Combine polls.jsonl + votes.jsonl → raw.json.

Usage:
  python pipeline/build_raw.py bundestag_2021_2025
  python pipeline/build_raw.py bundestag_2013_2017,bundestag_2017_2021,bundestag_2021_2025
"""
import json
import sys
from pathlib import Path

OUTPUT = Path(__file__).parent.parent / "output"

PERIODS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "bundestag_2005_2009", "bundestag_2009_2013", "bundestag_2013_2017",
    "bundestag_2017_2021", "bundestag_2021_2025", "bundestag_2025_2029",
]

for period in PERIODS:
    d = OUTPUT / period
    polls_path = d / "polls.jsonl"
    votes_path = d / "votes.jsonl"
    out_path   = d / "raw.json"

    if not polls_path.exists():
        print(f"  {period}: polls.jsonl not found, skipping")
        continue
    if not votes_path.exists():
        print(f"  {period}: votes.jsonl not found, skipping")
        continue

    polls = []
    with open(polls_path) as f:
        for line in f:
            line = line.strip()
            if line:
                polls.append(json.loads(line))

    votes = []
    with open(votes_path) as f:
        for line in f:
            line = line.strip()
            if line:
                votes.append(json.loads(line))

    with open(out_path, "w") as f:
        json.dump({"polls": polls, "votes": votes}, f, ensure_ascii=False)

    print(f"{period}: {len(polls)} polls, {len(votes)} votes → {out_path}")
