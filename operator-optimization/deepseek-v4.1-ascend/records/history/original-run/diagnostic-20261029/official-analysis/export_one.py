#!/usr/bin/env python3
"""Copy and offline-export one captured rank into this run's analysis tree."""
import os, pathlib, shutil, sys

run = pathlib.Path("/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official-20261030T084726Z-cb252d1d")
traces = run / "runtime" / "official-traces"
out = run / "analysis" / "operator-export"
name = sys.argv[1]
assert "/" not in name and name.endswith("_ascend_pt")
source, target = traces / name, out / "copies" / name
assert source.is_dir() and not target.exists(), "missing source or target exists"
tmp = out / "tmp"
tmp.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(tmp)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
print("COPY", name, flush=True)
shutil.copytree(source, target)
print("ANALYSE", name, flush=True)
from torch_npu.profiler.profiler import analyse
analyse(str(target), max_process_number=1, export_type="text")
print("DONE", name, flush=True)
