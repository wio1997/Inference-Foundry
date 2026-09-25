#!/usr/bin/env python3
"""Temporary no-barrier proposer wall/thread-CPU instrumentation."""
import argparse,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=[root/'bootstrap/vllm_dspark_handoff.py',root/'runtime/fixed_serving.py']
def sha(b):return hashlib.sha256(b).hexdigest()
def replace_once(s,old,new):
    assert s.count(old)==1,(old,s.count(old))
    return s.replace(old,new,1)
def patch_handoff(s):
    s=replace_once(s,'import os\nfrom contextlib import nullcontext', 'import os\nimport time\nfrom contextlib import nullcontext')
    s=replace_once(s,'        self.proposer.runner = _DP1RunnerShim(inputs)', '''        self.proposer.runner = _DP1RunnerShim(inputs)
        self._subphase_capture = os.getenv("EXTREME_PROPOSER_SUBPHASE_CAPTURE") == "1"
        self.subphase_rows = []
        self._subphase_current = None
        if self._subphase_capture:
            for name in ("set_inputs_first_pass", "_build_step_attn_metadatas", "_runnable", "_run_window_draft_steps"):
                original = getattr(self.proposer, name)
                def timed(*args, _name=name, _original=original, **kwargs):
                    wall0 = time.perf_counter_ns()
                    cpu0 = time.thread_time_ns()
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        row = self._subphase_current
                        if row is not None:
                            row["calls"].append({"name": _name, "wall_start_ns": wall0,
                                "wall_end_ns": time.perf_counter_ns(),
                                "thread_cpu_ns": time.thread_time_ns() - cpu0})
                setattr(self.proposer, name, timed)''')
    s=replace_once(s,'        markers: list[tuple[str, Any]] = []\n        def mark(label: str) -> None:', '''        markers: list[tuple[str, Any]] = []
        phase = {"cycle": state.cycle_index, "marks": [], "calls": []} if self._subphase_capture else None
        self._subphase_current = phase
        def mark(label: str) -> None:
            if phase is not None:
                phase["marks"].append({"name": label,
                    "wall_ns": time.perf_counter_ns(), "thread_cpu_ns": time.thread_time_ns()})''')
    s=replace_once(s,'        if markers:\n            self.dag_events.append(markers)\n        return next_draft', '''        if markers:
            self.dag_events.append(markers)
        if phase is not None:
            self.subphase_rows.append(phase)
            self._subphase_current = None
        return next_draft''')
    return s
def patch_serving(s):
    marker='        dag_dir = os.getenv("EXTREME_RUNTIME_DAG_DIR")'
    injected='''        subphase_dir = os.getenv("EXTREME_PROPOSER_SUBPHASE_DIR")
        if subphase_dir:
            rank = torch.distributed.get_rank()
            path = Path(subphase_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"rank{rank}.json").write_text(json.dumps({
                "rank": rank, "cycles": cycles,
                "clock": "host perf_counter_ns plus thread_time_ns; no extra NPU synchronization",
                "rows": self.runtime.proposer.subphase_rows,
            }))
'''
    return replace_once(s,marker,injected+marker)
parser=argparse.ArgumentParser();parser.add_argument('action',choices=('install','restore'));parser.add_argument('--record',required=True);a=parser.parse_args()
record=Path(a.record);record.parent.mkdir(parents=True,exist_ok=True)
if a.action=='install':
    assert not record.exists()
    originals=[p.read_bytes() for p in files]
    backups=[]
    for p,b in zip(files,originals):
        backup=record.parent/(p.name+'.run139.orig');backup.write_bytes(b);backups.append(str(backup))
    changed=[patch_handoff(originals[0].decode()).encode(),patch_serving(originals[1].decode()).encode()]
    for p,b in zip(files,changed):p.write_bytes(b)
    record.write_text(json.dumps({'paths':[str(p) for p in files],'backup_paths':backups,
       'original_sha256':[sha(b) for b in originals],'patched_sha256':[sha(b) for b in changed]},indent=2)+'\n')
else:
    assert record.exists();info=json.loads(record.read_text())
    for p,backup,expected in zip(files,info['backup_paths'],info['original_sha256']):
        b=Path(backup).read_bytes();assert sha(b)==expected;p.write_bytes(b)
    info['restored_sha256']=[sha(p.read_bytes()) for p in files]
    record.write_text(json.dumps(info,indent=2)+'\n')
print(record)
