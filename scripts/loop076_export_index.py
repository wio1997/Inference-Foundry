#!/usr/bin/env python3
"""Index exported CANN Level1 files for Run367 with a strict 8x5 gate."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile_dir = args.profile_dir.resolve()
    entries = []
    for rank in range(8):
        folders = sorted(profile_dir.glob(f"rank{rank}_*/ASCEND_PROFILER_OUTPUT"))
        assert len(folders) == 5, (rank, len(folders))
        for capture, folder in enumerate(folders):
            csv_path = folder / "kernel_details.csv"
            trace_path = folder / "trace_view.json"
            assert csv_path.is_file() and trace_path.is_file(), folder
            assert csv_path.stat().st_size > 1000000 and trace_path.stat().st_size > 1000000
            entries.append({
                "rank": rank, "capture": capture,
                "folder": str(folder.relative_to(ROOT)),
                "kernel_csv_bytes": csv_path.stat().st_size,
                "kernel_csv_sha256": hash_file(csv_path),
                "trace_bytes": trace_path.stat().st_size,
                "trace_sha256": hash_file(trace_path),
            })
    out = {"status": "all40_exports_present", "directories": len(entries), "rows": entries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "directories": len(entries)}))


if __name__ == "__main__":
    main()
