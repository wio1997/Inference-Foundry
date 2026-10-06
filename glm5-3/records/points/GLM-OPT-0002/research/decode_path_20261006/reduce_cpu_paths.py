"""Reproduce statistical path partitions from the saved, resolved Run254 raw.

This reads saved IP/callchain evidence only; it does not import torch or sample
the live model. Results are CPU sample contexts, never wall/saving budgets.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("resolved", type=Path)
args = parser.parse_args()
raw = args.resolved.read_bytes()
data = json.loads(raw)
frames = data["frame_table"]
counts = collections.Counter()
callers = collections.Counter()
per_pid = collections.defaultdict(collections.Counter)
kernel_samples = 0
kernel_leaf_samples = 0
for row in data["sample_refs"]:
    chain = [frames[i] for i in row["frame_ids"]]
    names = [f.get("demangled", "") for f in chain]
    kernel = next((i for i, n in enumerate(names)
                   if "PythonKernelHolder" in n), None)
    operators = [i for i, n in enumerate(names)
                 if "Dispatcher::callBoxed" in n
                 or "invokeOperatorFromPython" in n]
    getter = next((i for i, f in enumerate(chain)
                   if f.get("symbol") == "_Z16GetOpApiFuncAddrPKc"), None)
    if kernel is not None:
        kernel_samples += 1
    if "PythonKernelHolder" in frames[row["leaf_id"]].get("demangled", ""):
        kernel_leaf_samples += 1
    if getter is not None:
        category = "MC2_capability_query"
        # Actual predicate/call-site addresses were separately checked against
        # the installed binary. Other unnamed nearest-FDE values remain unnamed.
        assert chain[getter + 1]["fde_nearest_start"] == "0x490a9d0"
        site = chain[getter + 2]
        assert site["fde_nearest_start"] in ("0x31ed7c4", "0x31ba1d0")
        callers[(site["fde_nearest_start"], site["elf_va"])] += 1
    elif kernel is not None:
        category = ("nested_native_operator_under_Python_kernel"
                    if any(i < kernel for i in operators)
                    else "Python_kernel_body_or_callback_without_inner_boxed_operator")
    else:
        category = ("operator_call_without_sampled_Python_kernel_frame"
                    if operators else "other_user_CPU_unknown_Python_function")
    counts[category] += 1
    per_pid[row["pid"]][category] += 1

assert sum(counts.values()) == data["event_count"] == len(data["sample_refs"])
out = dict(
    run_id=data["run_id"], raw_sha256=data["raw_sha256"],
    resolved_sha256=hashlib.sha256(raw).hexdigest(),
    event_count=data["event_count"], lost=data["lost"],
    partitions=[dict(category=k, samples=v, share=v/data["event_count"])
                for k, v in counts.most_common()],
    kernel_inclusive_samples=kernel_samples,
    kernel_leaf_samples=kernel_leaf_samples,
    capability_callers=[dict(caller_fde=k[0], return_pc=k[1], samples=v)
                        for k, v in callers.most_common()],
    per_pid={str(k): dict(v) for k, v in sorted(per_pid.items())},
    limits=[
        "Disjoint statistical call-stack contexts, not wall budgets or removable fractions.",
        "A Python handler can call native code without an inner boxed-operator frame.",
        "No Python source frames or Scheduler/Executor/ModelRunner entry/exit markers.",
        "Inclusive PythonKernelHolder includes the handler body and inner native operators.",
        "Per-rank CPU shares do not establish that rank's criticality.",
    ],
)
print(json.dumps(out, indent=2))
