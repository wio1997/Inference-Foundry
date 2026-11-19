"""Read rank-zero Decode prefix markers from one immutable profiler artifact."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3


PREFIX = "ds41.profile.decode_prefix_commit "
MARKER = re.compile(
    rf"{re.escape(PREFIX)}room=([a-f0-9]{{16}}|unavailable) prompt=([0-9]+) "
    r"rebootstrap=([01]) match=(?:unavailable|known l1=([0-9]+) "
    r"l2=([0-9]+) l3=([0-9]+) declared=([0-9]+))"
)


def marker_names(path, source_format):
    if source_format == "chrome":
        with path.open() as stream:
            events = json.load(stream)["traceEvents"]
        for event in events:
            name = event.get("name")
            if event.get("ph") == "X" and isinstance(name, str) and name.startswith(PREFIX):
                yield name
        return

    for suffix in ("-wal", "-journal"):
        if Path(f"{path}{suffix}").exists():
            raise ValueError(f"SQLite artifact has a {suffix} sidecar")
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True) as db:
        for name, in db.execute(
            "SELECT names.value FROM PYTORCH_API AS api "
            "JOIN STRING_IDS AS names ON api.name = names.id "
            "WHERE substr(names.value, 1, ?) = ?",
            (len(PREFIX), PREFIX),
        ):
            yield name


def audit(names, requests, prompt_tokens):
    observed = Counter()
    eligible = {}
    totals = [0, 0, 0, 0]
    for name in names:
        observed["markers"] += 1
        marker = MARKER.fullmatch(name)
        if marker is None:
            observed["malformed"] += 1
            continue
        room, prompt, rebootstrap, l1, l2, l3, declared = marker.groups()
        counts = tuple(map(int, (l1, l2, l3, declared))) if l1 is not None else None
        if counts is not None and (sum(counts[:3]) != counts[3] or counts[3] > int(prompt)):
            observed["malformed"] += 1
        elif rebootstrap == "1":
            observed["rebootstrap"] += 1
        elif room == "unavailable":
            observed["unavailable_room"] += 1
        elif int(prompt) != prompt_tokens:
            observed["unexpected_prompt"] += 1
        elif room in eligible:
            observed["duplicate_room"] += 1
        else:
            eligible[room] = counts
            if counts is None:
                observed["unavailable_match"] += 1
                continue
            observed["known_rooms"] += 1
            if counts[3] == 0:
                observed["zero_match_rooms"] += 1
            for index, count in enumerate(counts):
                totals[index] += count
    full_coverage = (
        observed["markers"] == requests + observed["rebootstrap"]
        and observed["known_rooms"] == requests
    )
    return {
        "expected_requests": requests,
        "expected_prompt_tokens": prompt_tokens,
        "fresh_expected_prompt_rooms": len(eligible),
        **{key: observed[key] for key in (
            "markers", "malformed", "rebootstrap", "unavailable_room", "unexpected_prompt",
            "duplicate_room", "unavailable_match", "known_rooms", "zero_match_rooms",
        )},
        "full_coverage": full_coverage,
        "known_unique_fresh_room_token_totals": dict(zip(("l1", "l2", "l3", "declared"), totals)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--format", required=True, choices=("chrome", "ascend-sqlite"))
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--requests", required=True, type=int)
    parser.add_argument("--prompt-tokens", required=True, type=int)
    args = parser.parse_args()
    if min(args.requests, args.prompt_tokens) < 1:
        parser.error("requests and prompt tokens must be positive")
    digest = sha256()
    with args.artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != args.sha256:
        parser.error("artifact SHA256 mismatch")
    report = audit(marker_names(args.artifact, args.format), args.requests, args.prompt_tokens)
    print(json.dumps({"artifact_SHA256": args.sha256, "format": args.format, **report}, sort_keys=True))


if __name__ == "__main__":
    main()
