"""Join Decode trace markers to successful raw AISBench requests by full prompt."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from audit_decode_prefix_trace_g1389 import MARKER, audit, marker_names


DIGEST = re.compile(r"[a-f0-9]{64}")
PROMPT_FIELD = " prompt_ids_sha256="


def verified_bytes(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch for {path.name}")
    return raw


def verified_requests(frozen, raw_requests, prompt_tokens, output_tokens):
    records = frozen["records"]
    if not records:
        raise ValueError("Expected a nonempty frozen request set")
    by_data = {}
    input_hashes = set()
    for record in records:
        index = record["data_id"]
        digest = record["token_ids_sha256"]
        if (type(index) is not int or index in by_data or record["input_tokens"] != prompt_tokens
                or not DIGEST.fullmatch(digest) or digest in input_hashes):
            raise ValueError("Invalid or duplicate frozen request identity")
        by_data[index] = record
        input_hashes.add(digest)
    if set(by_data) != set(range(len(records))):
        raise ValueError("Frozen request indexes must cover one full sequence")

    by_digest = {}
    request_ids = set()
    for row in raw_requests:
        record = by_data.get(row["data_id"])
        text = row["input"]
        if (record is None or type(row["data_id"]) is not int or row["id"] in request_ids
                or row["success"] is not True or row["input_tokens"] != prompt_tokens
                or row["output_tokens"] != output_tokens or not isinstance(text, str)
                or sha256(text.encode()).hexdigest() != record["question_sha256"]):
            raise ValueError("Raw AISBench request failed identity, status or token-count checks")
        digest = record["token_ids_sha256"]
        if digest in by_digest:
            raise ValueError("Raw AISBench request repeated a frozen data index")
        request_ids.add(row["id"])
        by_digest[digest] = {"request_id": row["id"], "data_id": row["data_id"]}
    if set(by_digest) != input_hashes:
        raise ValueError("Raw AISBench requests do not cover the frozen prompts")
    return by_digest


def bind(names, expected, prompt_tokens):
    base_names = []
    bound = {}
    unbound = 0
    duplicate_prompt = 0
    for name in names:
        base, delimiter, digest = name.rpartition(PROMPT_FIELD)
        base_names.append(base if delimiter else name)
        marker = MARKER.fullmatch(base)
        if marker is None or not DIGEST.fullmatch(digest) or digest not in expected:
            unbound += 1
            continue
        room, prompt, rebootstrap, l1, l2, l3, declared = marker.groups()
        if rebootstrap == "1" or room == "unavailable" or int(prompt) != prompt_tokens:
            continue
        if digest in bound:
            duplicate_prompt += 1
            continue
        counts = None if l1 is None else dict(zip(
            ("l1", "l2", "l3", "declared"), map(int, (l1, l2, l3, declared))
        ))
        bound[digest] = {**expected[digest], "room_fingerprint": room,
                         "prompt_ids_sha256": digest, "prefix_match": counts}
    trace = audit(base_names, len(expected), prompt_tokens)
    complete = (trace["full_coverage"] and not unbound and not duplicate_prompt
                and set(bound) == set(expected))
    return {
        "all_requests_bound_with_known_match": complete,
        "unbound_markers": unbound,
        "duplicate_fresh_prompt_markers": duplicate_prompt,
        "trace_audit": trace,
        "bound_requests": sorted(bound.values(), key=lambda row: row["data_id"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--format", required=True, choices=("chrome", "ascend-sqlite"))
    parser.add_argument("--trace-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--details-sha256", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=32768)
    parser.add_argument("--output-tokens", type=int, default=1024)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.output_tokens) <= 0:
        parser.error("Expected positive token counts")
    with args.trace.open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != args.trace_sha256:
        parser.error("Trace SHA256 mismatch")
    frozen = json.loads(verified_bytes(args.manifest, args.manifest_sha256))
    details = verified_bytes(args.details, args.details_sha256)
    raw_requests = [json.loads(row) for row in details.splitlines() if row.strip()]
    expected = verified_requests(frozen, raw_requests, args.prompt_tokens, args.output_tokens)
    report = bind(marker_names(args.trace, args.format), expected, args.prompt_tokens)
    print(json.dumps({
        "trace_sha256": args.trace_sha256, "trace_format": args.format,
        "manifest_sha256": args.manifest_sha256, "dataset_sha256": frozen["dataset_sha256"],
        "details_sha256": args.details_sha256, **report,
    }, sort_keys=True))
    return 0 if report["all_requests_bound_with_known_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
