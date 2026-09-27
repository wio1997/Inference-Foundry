#!/usr/bin/env python3
"""Reversible, SHA-pinned Host-only Scheduler bulk-ledger patch."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
SOURCE = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/patch/platform/patch_kv_delivery_preemption.py")
EXPECTED = "5ba894b9ec003502c008e12281c260787ed64a419327c451c8e67374bf525e8d"
MARK = "# EXTREME_LOOP079_BULK_LEDGER"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def patch(source):
    if MARK in source:
        raise ValueError("already patched")
    begin = "                if model_runner_output.extreme_bulk_output:\n"
    assert source.count(begin) == 1
    source = source.replace(begin, begin + """                    # EXTREME_LOOP079_BULK_LEDGER
                    from scripts import loop079_bulk_ledger as _bulk_ledger
                    _bulk_row = _bulk_ledger.before(self, request, new_token_ids, output_is_stale)
""")
    end = """                            self, request, new_token_ids
                        )
                    )
                else:
"""
    assert source.count(end) == 1
    return source.replace(end, """                            self, request, new_token_ids
                        )
                    )
                    # EXTREME_LOOP079_BULK_LEDGER
                    _bulk_ledger.after(_bulk_row, request, new_token_ids, stopped)
                else:
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("check", "install", "restore"))
    ap.add_argument("--state-dir", type=Path)
    ap.add_argument("--record", type=Path, required=True)
    a = ap.parse_args()
    if a.action in ("check", "install"):
        old = SOURCE.read_bytes()
        if digest(old) != EXPECTED:
            raise ValueError("borrowed Scheduler source SHA changed")
        new = patch(old.decode()).encode()
        compile(new, str(SOURCE), "exec")
        helper = ROOT / "scripts/loop079_bulk_ledger.py"
        compile(helper.read_bytes(), str(helper), "exec")
        report = {"action": a.action, "source": str(SOURCE), "original": digest(old), "patched": digest(new), "helper": digest(helper.read_bytes())}
        if a.action == "install":
            if a.state_dir is None or a.state_dir.exists():
                raise ValueError("install requires fresh state-dir")
            a.state_dir.mkdir(parents=True)
            (a.state_dir / "original.py").write_bytes(old)
            (a.state_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
            if SOURCE.read_bytes() != old:
                raise ValueError("source changed during install")
            SOURCE.write_bytes(new)
    else:
        if a.state_dir is None:
            raise ValueError("restore requires state-dir")
        report = json.loads((a.state_dir / "manifest.json").read_text())
        old = (a.state_dir / "original.py").read_bytes()
        if report["source"] != str(SOURCE) or report["original"] != digest(old) or digest(old) != EXPECTED:
            raise ValueError("backup/manifest mismatch")
        if digest(SOURCE.read_bytes()) != report["patched"]:
            raise ValueError("patched source changed; refusing restore")
        SOURCE.write_bytes(old)
        report = dict(report, action="restore", restored=digest(SOURCE.read_bytes()) == EXPECTED)
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"action": a.action, "original": EXPECTED, "patched": report["patched"]}))


if __name__ == "__main__":
    main()
