#!/usr/bin/env python3
"""Run fixed per-rank offline exports, with at most two jobs."""
import concurrent.futures, pathlib, subprocess, sys

p = pathlib.Path(__file__).resolve().parent
names = (p / "rank-directories.txt").read_text().splitlines()
assert len(names) == 32 and len(set(names)) == 32
first = "dp0_pp0_tp0_dcp0_ep0_rank0_758_20261030085235356_ascend_pt"
assert first in names and "DONE " + first in (p / (first + ".stdout")).read_text()
names.remove(first)
def run(name):
    c = subprocess.run([sys.executable, str(p / "run_export_one.py"), name], check=False)
    return name, c.returncode
failed = []
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for name, rc in pool.map(run, names):
        print(name, rc, flush=True)
        if rc:
            failed.append((name, rc))
print("FAILURES", failed, flush=True)
sys.exit(bool(failed))
