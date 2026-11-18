"""Independently recompute AISBench timing from its original NumPy SQLite blobs.

This reads measurements and never generates load or modifies original evidence.
NumPy files are decoded with the standard library; pickles are not accepted.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import struct


def npy_float64(blob):
    assert blob[:6] == b"\x93NUMPY"
    version = tuple(blob[6:8])
    if version == (1, 0):
        offset = 10
        size = struct.unpack("<H", blob[8:10])[0]
    else:
        assert version in ((2, 0), (3, 0))
        offset = 12
        size = struct.unpack("<I", blob[8:12])[0]
    header = ast.literal_eval(blob[offset:offset + size].decode())
    assert header["descr"] == "<f8" and header["fortran_order"] is False
    assert len(header["shape"]) == 1
    count = header["shape"][0]
    assert 3 <= count <= 100000
    payload = blob[offset + size:]
    assert len(payload) == count * 8
    values = struct.unpack("<" + str(count) + "d", payload)
    assert all(math.isfinite(x) for x in values)
    assert all(a <= b for a, b in zip(values, values[1:]))
    assert values[-1] > values[1] >= values[0]
    return values


def audit_phase(run_root, phase, length):
    run_root = Path(run_root)
    phase_root = run_root / f"{phase}_{length}"
    paths = list(phase_root.rglob("*_details.jsonl"))
    assert len(paths) == 1
    details = paths[0]
    rows = [json.loads(line) for line in details.read_text().splitlines()]
    assert len(rows) == 32 and len({x["data_id"] for x in rows}) == 32
    assert all(x["success"] is True and not x["error_info"] for x in rows)
    assert all(x["input_tokens"] == length and x["output_tokens"] == 1024 for x in rows)
    arrays = []
    db_hashes = {}
    refs = set()
    for row in rows:
        name = row["db_name"]
        assert Path(name).name == name
        db = details.parent / "db_data" / name
        ref = row["time_points"]
        assert set(ref) == {"__db_ref__"} and isinstance(ref["__db_ref__"], int)
        key = (name, ref["__db_ref__"])
        assert key not in refs
        refs.add(key)
        with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            record = connection.execute(
                "SELECT arr_blob FROM numpy_store WHERE id=?", (ref["__db_ref__"],)
            ).fetchone()
            assert record is not None
        arrays.append(npy_float64(record[0]))
        db_hashes[name] = hashlib.sha256(db.read_bytes()).hexdigest()
    duration = max(t[-1] for t in arrays) - min(t[0] for t in arrays)
    ttft = statistics.mean((t[1] - t[0]) * 1000 for t in arrays)
    e2el = statistics.mean((t[-1] - t[0]) * 1000 for t in arrays)
    tpot = statistics.mean((t[-1] - t[1]) * 1000 / 1023 for t in arrays)
    active = peak = 0
    for _, delta in sorted([(t[0], 1) for t in arrays] + [(t[-1], -1) for t in arrays]):
        active += delta
        peak = max(active, peak)
    assert active == 0 and peak == 16
    source = json.loads((run_root / f"{phase}_{length}_audit.json").read_text())
    assert math.isclose(duration, source["duration_s"], abs_tol=1e-6, rel_tol=0)
    for name, value in [("ttft_mean_ms", ttft), ("e2el_mean_ms", e2el), ("tpot_mean_ms", tpot)]:
        # CSV reports one decimal place in milliseconds.
        assert abs(value - source[name]) <= 0.050001, (name, value, source[name])
    for name, total in [("input_tokens_per_s", 32 * length),
                        ("output_tokens_per_s", 32 * 1024),
                        ("total_tokens_per_s", 32 * (length + 1024))]:
        assert math.isclose(total / duration, source[name], rel_tol=1e-5, abs_tol=0.01)
    return {
        "phase": phase, "requests": len(rows), "input_tokens": length,
        "output_tokens": 1024, "observed_peak_from_raw_intervals": peak,
        "recomputed_duration_s": duration, "recomputed_ttft_mean_ms": ttft,
        "recomputed_e2el_mean_ms": e2el, "recomputed_tpot_mean_ms": tpot,
        "database_sha256": db_hashes,
        "all_raw_request_counts_and_reported_metrics_match": True,
        "scope": "AISBench time-point convention and reported precision; no accuracy or cache-hit claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {phase: audit_phase(args.run_root, phase, args.length) for phase in ("cold", "hot")}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
