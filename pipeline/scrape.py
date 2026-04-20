#!/usr/bin/env python3
"""
Download Bundestag roll-call voting data from abgeordnetenwatch API v2.

Outputs:
  - polls.jsonl  : one poll per line
  - votes.jsonl  : one vote per line (MP mandate vote), includes _poll_id for joining
  - checkpoint.json : progress for resuming

Usage examples:
  python scrape.py
  python scrape.py --legislature 111
  python scrape.py --legislature 111 --pager-limit 200
  python scrape.py --since 2019-01-01 --until 2021-12-31 --legislature 111
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import requests

BASE = "https://www.abgeordnetenwatch.de/api/v2"
DEFAULT_PAGER_LIMIT = 100  # safe default per docs


# ---------------------------
# HTTP + API helpers
# ---------------------------


def _sleep_backoff(attempt: int) -> None:
    # exponential-ish backoff: 0.5, 1, 2, 4, 8...
    time.sleep(min(8.0, 0.5 * (2**attempt)))


def _peek_text(resp: requests.Response, limit: int = 800) -> str:
    txt = resp.text.replace("\n", " ")
    return txt[:limit] + ("..." if len(txt) > limit else "")


class APIError(RuntimeError):
    pass


def api_get_json(
    session: requests.Session,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    max_retries: int = 5,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    GET an API endpoint and return parsed JSON.

    Enforces:
      - HTTP 200
      - JSON parseability
      - meta.status == "ok" (otherwise raises APIError with message)
    """
    url = f"{BASE}{path}"
    params = params or {}

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)

            if resp.status_code != 200:
                raise APIError(
                    f"HTTP {resp.status_code} for {url} params={params}\n"
                    f"Preview: {_peek_text(resp)}"
                )

            try:
                data = resp.json()
            except Exception as e:
                raise APIError(
                    f"Non-JSON response for {url} params={params} "
                    f"(content-type={resp.headers.get('content-type')}).\n"
                    f"Preview: {_peek_text(resp)}"
                ) from e

            meta = data.get("meta")
            if not isinstance(meta, dict):
                raise APIError(
                    f"Missing/invalid meta for {url} params={params}: {str(data)[:800]}..."
                )

            status = meta.get("status")
            if status != "ok":
                # docs: if meta.status == error, data is empty; meta.message contains details
                msg = meta.get("message") or meta.get("messages") or "(no message)"
                raise APIError(
                    f"API error for {url} params={params}: meta.status={status} msg={msg}"
                )

            # data should exist even if empty list
            if "data" not in data:
                raise APIError(
                    f"Missing data field for {url} params={params}: {str(data)[:800]}..."
                )

            return data

        except (requests.RequestException, APIError) as e:
            last_exc = e
            # Retry on transient network issues; APIError could be transient too.
            if attempt < max_retries - 1:
                _sleep_backoff(attempt)
                continue
            raise

    raise APIError(f"Failed to fetch {url} after retries: {last_exc}")


def extract_list_data(
    payload: Dict[str, Any], expected_key: str
) -> List[Dict[str, Any]]:
    """
    For list endpoints, docs say:
      - top-level: {"meta": ..., "data": [ ... ]}  OR sometimes {"data": {"polls": [ ... ]}}
    The documentation indicates data is list/object depending on endpoint.
    In practice, many endpoints use {"data": { "<key>": [ ... ] }}.

    We support both:
      - if payload["data"] is list: return it
      - if payload["data"] is dict containing expected_key list: return it
    """
    data = payload.get("data")

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        val = data.get(expected_key)
        if isinstance(val, list):
            return val

    raise APIError(
        f"Unexpected data shape. expected list or data['{expected_key}'] list.\n"
        f"Top keys: {list(payload.keys())}; data type: {type(data)}; preview: {str(payload)[:800]}..."
    )


def extract_object_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    For single-entity endpoints, docs say data is an object.
    """
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    raise APIError(
        f"Unexpected single-entity data shape: data type={type(data)} preview={str(payload)[:800]}..."
    )


def meta_pages(payload: Dict[str, Any]) -> Optional[int]:
    """
    Try to read total pages from meta.result.pages.
    Docs show meta.result with counters.
    We'll handle missing gracefully.
    """
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        return None
    result = meta.get("result", {})
    if not isinstance(result, dict):
        return None
    pages = result.get("pages")
    if pages is None:
        return None
    try:
        return int(pages)
    except Exception:
        return None


def meta_total(payload: Dict[str, Any]) -> Optional[int]:
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        return None
    result = meta.get("result", {})
    if not isinstance(result, dict):
        return None
    total = result.get("total")
    if total is None:
        total = result.get("count")
    if total is None:
        return None
    try:
        return int(total)
    except Exception:
        return None


# ---------------------------
# Pagination iterators
# ---------------------------


def iter_collection(
    session: requests.Session,
    path: str,
    *,
    expected_key: str,
    params: Optional[Dict[str, Any]] = None,
    pager_limit: int = DEFAULT_PAGER_LIMIT,
    start_page: int = 0,
    polite_sleep: float = 0.15,
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """
    Iterate items of a collection endpoint using page/pager_limit.

    Yields (page_number, item_dict).

    Stops when:
      - meta.result.pages is reached (if present), else
      - returned items < pager_limit
    """
    page = start_page
    params = dict(params or {})
    params["pager_limit"] = pager_limit

    while True:
        params["page"] = page
        payload = api_get_json(session, path, params=params)
        items = extract_list_data(payload, expected_key=expected_key)

        for item in items:
            yield page, item

        pages = meta_pages(payload)
        if pages is not None:
            if page + 1 >= pages:
                return
        else:
            if len(items) < pager_limit:
                return

        page += 1
        time.sleep(polite_sleep)


# ---------------------------
# Checkpointing
# ---------------------------


def load_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save_checkpoint(path: str, ckpt: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------
# Domain-specific fetchers
# ---------------------------


def poll_params(
    legislature_id: Optional[int], since: Optional[str], until: Optional[str]
) -> Dict[str, Any]:
    """
    Build poll filters.
    Dates follow YYYY-MM-DD. The Poll entity uses field_poll_date.
    Operators are supported via [gte]/[lte] per docs.
    """
    p: Dict[str, Any] = {}
    if legislature_id is not None:
        p["field_legislature"] = legislature_id
    if since:
        p["field_poll_date[gte]"] = since
    if until:
        p["field_poll_date[lte]"] = until
    return p


def iter_polls(
    session: requests.Session,
    *,
    legislature_id: Optional[int],
    since: Optional[str],
    until: Optional[str],
    pager_limit: int,
    start_page: int = 0,
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    return iter_collection(
        session,
        "/polls",
        expected_key="polls",
        params=poll_params(legislature_id, since, until),
        pager_limit=pager_limit,
        start_page=start_page,
    )


def iter_votes_for_poll(
    session: requests.Session,
    poll_id: int,
    *,
    pager_limit: int,
) -> Iterator[Dict[str, Any]]:
    # Vote entity supports filter poll=<id>
    for _, item in iter_collection(
        session,
        "/votes",
        expected_key="votes",
        params={"poll": poll_id},
        pager_limit=pager_limit,
        start_page=0,
        polite_sleep=0.10,
    ):
        yield item


# ---------------------------
# Main scraping routine
# ---------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--legislature",
        type=int,
        default=None,
        help="ParliamentPeriod id (e.g. 111 for 19. WP).",
    )
    ap.add_argument(
        "--since",
        type=str,
        default=None,
        help="YYYY-MM-DD (inclusive) filter on poll date.",
    )
    ap.add_argument(
        "--until",
        type=str,
        default=None,
        help="YYYY-MM-DD (inclusive) filter on poll date.",
    )
    ap.add_argument(
        "--pager-limit",
        type=int,
        default=DEFAULT_PAGER_LIMIT,
        help="1..1000 (default 100).",
    )
    ap.add_argument("--outdir", type=str, default=".", help="Output directory.")
    ap.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint.json if present."
    )
    args = ap.parse_args()

    pager_limit = int(args.pager_limit)
    if pager_limit < 1 or pager_limit > 1000:
        raise SystemExit("--pager-limit must be between 1 and 1000")

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    polls_path = os.path.join(outdir, "polls.jsonl")
    votes_path = os.path.join(outdir, "votes.jsonl")
    ckpt_path = os.path.join(outdir, "checkpoint.json")

    session = requests.Session()
    session.headers.update({"User-Agent": "bundestag-votes-research/1.0"})

    ckpt = load_checkpoint(ckpt_path) if args.resume else {}
    start_poll_page = int(ckpt.get("poll_page", 0))
    processed_poll_ids = set(
        ckpt.get("processed_poll_ids", [])
    )  # to avoid duplicates on resume

    print(f"Output dir: {outdir}")
    print(f"Writing: {polls_path}")
    print(f"Writing: {votes_path}")
    print(
        f"Resume: {bool(args.resume)} (start poll page: {start_poll_page}, already processed polls: {len(processed_poll_ids)})"
    )

    # Open in append mode so resume can continue.
    with open(polls_path, "a", encoding="utf-8") as fpolls, open(
        votes_path, "a", encoding="utf-8"
    ) as fvotes:
        last_page_seen = start_poll_page

        for page, poll in iter_polls(
            session,
            legislature_id=args.legislature,
            since=args.since,
            until=args.until,
            pager_limit=pager_limit,
            start_page=start_poll_page,
        ):
            last_page_seen = page

            poll_id = poll.get("id")
            if poll_id is None:
                # Shouldn't happen, but don't crash.
                continue

            if poll_id in processed_poll_ids:
                # already done on resume
                continue

            # write poll record
            fpolls.write(json.dumps(poll, ensure_ascii=False) + "\n")
            fpolls.flush()

            # fetch and write votes for this poll
            votes_written = 0
            for vote in iter_votes_for_poll(
                session, int(poll_id), pager_limit=pager_limit
            ):
                vote["_poll_id"] = int(poll_id)
                fvotes.write(json.dumps(vote, ensure_ascii=False) + "\n")
                votes_written += 1

            fvotes.flush()

            processed_poll_ids.add(int(poll_id))

            # checkpoint every poll
            ckpt = {
                "poll_page": last_page_seen,  # safe: resume might re-read this page, but we dedupe by poll_id
                "processed_poll_ids": sorted(processed_poll_ids),
                "legislature": args.legislature,
                "since": args.since,
                "until": args.until,
                "pager_limit": pager_limit,
                "updated_at_unix": int(time.time()),
            }
            save_checkpoint(ckpt_path, ckpt)

            print(f"poll_id={poll_id}  votes={votes_written}  page={page}")

            time.sleep(0.20)  # politeness between polls


if __name__ == "__main__":
    main()
