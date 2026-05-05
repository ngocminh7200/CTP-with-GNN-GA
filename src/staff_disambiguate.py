"""
Interactive staff disambiguation tool.

Run after preprocess.py:
    python src/staff_disambiguate.py

Steps:
  1. Loads the current staff_links_df.
  2. Finds candidate match pairs by similarity.
  3. Asks you yes/no for each candidate.
  4. Saves your decisions to data/processed/staff_aliases.json.

An old file - no longer needed, the excel space fix actually solved the duplicacy issue
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from config import PROCESSED_DIR
from timetable_data import TimetableData

ALIAS_FILE = PROCESSED_DIR / "staff_aliases.json"

_TITLES = {"DR", "MR", "MRS", "MS", "MISS", "PROF", "PROFESSOR"}


def parse_name(raw: str) -> dict:
    """Return {'last', 'titles', 'first', 'first_initial'} for a 'LAST, TITLE FIRST' string."""
    if "," not in raw:
        return {"last": raw.strip(), "titles": [], "first": "", "first_initial": ""}
    last, rest = raw.split(",", 1)
    toks = rest.strip().split()
    titles = [t for t in toks if t in _TITLES]
    others = [t for t in toks if t not in _TITLES]
    first = " ".join(others)
    first_init = others[0][0] if others and others[0] else ""
    return {
        "last": last.strip(),
        "titles": titles,
        "first": first,
        "first_initial": first_init,
    }


def candidate_score(a: str, b: str) -> tuple[float, list[str]]:
    """Return (similarity score in [0,1], list of reasons) for raw strings a, b."""
    pa, pb = parse_name(a), parse_name(b)
    reasons = []
    score = 0.0

    # Last name match
    if pa["last"] and pa["last"] == pb["last"]:
        score += 0.4
        reasons.append("same last name")
    elif pa["last"] and pb["last"] and SequenceMatcher(None, pa["last"], pb["last"]).ratio() >= 0.85:
        score += 0.2
        reasons.append("similar last name")

    # First initial match
    if pa["first_initial"] and pa["first_initial"] == pb["first_initial"]:
        score += 0.2
        reasons.append("same first initial")

    # Full-first-name match
    if pa["first"] and pb["first"] and pa["first"] == pb["first"]:
        score += 0.3
        reasons.append("same full first name")
    elif pa["first"] and pb["first"] and SequenceMatcher(None, pa["first"], pb["first"]).ratio() >= 0.8:
        score += 0.15
        reasons.append("similar first name")

    # One side has no first name (likely abbreviation case)
    if (not pa["first"]) ^ (not pb["first"]):
        score += 0.1
        reasons.append("one side missing first name")

    # Overall string similarity
    overall = SequenceMatcher(None, a, b).ratio()
    score += 0.1 * overall

    return score, reasons


def find_candidates(staff_ids: list[str], threshold: float = 0.55) -> list[tuple[str, str, float, list[str]]]:
    """Find all pairs of staff_ids whose similarity score exceeds threshold.

    Optimization: bucket by last name first to avoid the O(n²) full sweep.
    """
    by_last = defaultdict(list)
    for s in staff_ids:
        p = parse_name(s)
        by_last[p["last"]].append(s)

    candidates = []
    for last, group in by_last.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                score, reasons = candidate_score(a, b)
                if score >= threshold:
                    candidates.append((a, b, score, reasons))

    candidates.sort(key=lambda x: -x[2])
    return candidates


def load_existing_aliases() -> dict[str, list[str]]:
    if ALIAS_FILE.exists():
        with ALIAS_FILE.open() as f:
            return json.load(f)
    return {}


def save_aliases(aliases: dict[str, list[str]]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with ALIAS_FILE.open("w") as f:
        json.dump(aliases, f, indent=2, sort_keys=True)


def aliases_to_lookup(aliases: dict[str, list[str]]) -> dict[str, str]:
    """Invert the aliases dict to {raw_name: canonical_name}."""
    lookup = {}
    for canonical, raws in aliases.items():
        for r in raws:
            lookup[r] = canonical
    return lookup


def make_canonical(a: str, b: str) -> str:
    """Default canonical = the longer/more-informative of the two."""
    pa, pb = parse_name(a), parse_name(b)
    # Prefer the one with a full first name
    if pa["first"] and not pb["first"]:
        return f"{pa['last']} {pa['first']}".upper()
    if pb["first"] and not pa["first"]:
        return f"{pb['last']} {pb['first']}".upper()
    # Otherwise pick the longer string
    longer = a if len(a) >= len(b) else b
    p = parse_name(longer)
    return f"{p['last']} {p['first']}".strip().upper() or longer


def main() -> None:
    print("Loading data...")
    data = TimetableData.load()
    staff_ids = data.staff["staff_id"].tolist()
    print(f"Distinct raw staff identifiers: {len(staff_ids)}\n")

    aliases = load_existing_aliases()
    lookup = aliases_to_lookup(aliases)
    print(f"Loaded {sum(len(v) for v in aliases.values())} existing aliases "
          f"across {len(aliases)} canonical names.\n")

    print("Finding candidate matches (this may take a moment)...")
    candidates = find_candidates(staff_ids)
    print(f"Found {len(candidates)} candidate pairs above the threshold.\n")

    # Filter out pairs already resolved (both already mapped to the same canonical)
    pending = []
    for a, b, score, reasons in candidates:
        ca, cb = lookup.get(a, a), lookup.get(b, b)
        if ca == cb:
            continue   # already merged
        pending.append((a, b, score, reasons))

    print(f"{len(pending)} pairs still need decisions.\n")
    print("For each pair:  y = merge   n = skip   q = quit & save\n")
    print("=" * 60)

    decisions_made = 0
    for idx, (a, b, score, reasons) in enumerate(pending):
        print(f"\n[{idx + 1}/{len(pending)}] score={score:.2f} ({', '.join(reasons)})")
        print(f"  A: {a}")
        print(f"  B: {b}")
        suggested = make_canonical(a, b)
        print(f"  → canonical (default): {suggested}")
        ans = input("  merge? [y/n/q/edit-canonical] ").strip().lower()

        if ans == "q":
            break
        if ans == "n" or ans == "":
            continue

        # Allow user to override the canonical
        if ans not in {"y", "yes"}:
            canonical = ans.upper()
        else:
            canonical = suggested

        # Update aliases dict
        existing_a = lookup.get(a)
        existing_b = lookup.get(b)

        # Decide which canonical to use
        if existing_a and existing_b and existing_a != existing_b:
            print(f"  ⚠ conflict: A is currently '{existing_a}', B is '{existing_b}'. Skipping.")
            continue
        target = existing_a or existing_b or canonical

        if target not in aliases:
            aliases[target] = []
        for raw in (a, b):
            if raw not in aliases[target]:
                aliases[target].append(raw)
            lookup[raw] = target

        decisions_made += 1

    save_aliases(aliases)
    print(f"\nSaved {decisions_made} new decisions to {ALIAS_FILE}")
    print(f"Total canonical names: {len(aliases)} | total aliased raws: {sum(len(v) for v in aliases.values())}")


if __name__ == "__main__":
    main()