#!/usr/bin/env python3
"""Install/restore opt-in host phase timestamps for one diagnostic run."""
import argparse,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=[root/'runtime/extreme_decode.py',root/'runtime/fixed_serving.py']
def sha(b):return hashlib.sha256(b).hexdigest()
def replace_once(s,old,new):
    assert s.count(old)==1,(old,s.count(old))
    return s.replace(old,new,1)
def patch_decode(s):
    s=replace_once(s,'        self._profile_dag = os.getenv("EXTREME_RUNTIME_PROFILE_DAG") == "1"',
       '        self._profile_dag = os.getenv("EXTREME_RUNTIME_PROFILE_DAG") == "1"\n        self._phase_capture = os.getenv("EXTREME_RUNTIME_PHASE_CAPTURE") == "1"\n        self.phase_rows = []')
    s=replace_once(s,'        markers = []\n        def mark(label):',
       '        markers = []\n        phase_marks = []\n        def mark(label):\n            if self._phase_capture:\n                phase_marks.append((label, time.time_ns()))')
    s=replace_once(s,'            if markers:\n                self.diagnostic_events.append(markers)',
       '            if markers:\n                self.diagnostic_events.append(markers)\n            if self._phase_capture:\n                self.phase_rows.append({"cycle": self.state.cycle_index - 1, "marks": phase_marks})')
    return s
def patch_serving(s):
    marker='        dag_dir = os.getenv("EXTREME_RUNTIME_DAG_DIR")'
    injected='''        phase_dir = os.getenv("EXTREME_RUNTIME_PHASE_DIR")
        if phase_dir:
            rank = torch.distributed.get_rank()
            path = Path(phase_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"rank{rank}.json").write_text(json.dumps({
                "rank": rank, "cycles": cycles,
                "clock": "host time.time_ns, no additional device sync",
                "phase_rows": self.runtime.phase_rows,
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
        backup=record.parent/(p.name+'.run137.orig')
        backup.write_bytes(b);backups.append(str(backup))
    changed=[patch_decode(originals[0].decode()).encode(),patch_serving(originals[1].decode()).encode()]
    for p,b in zip(files,changed):p.write_bytes(b)
    record.write_text(json.dumps({'paths':[str(p) for p in files],'backup_paths':backups,
       'original_sha256':[sha(b) for b in originals],'patched_sha256':[sha(b) for b in changed]},indent=2)+'\n')
else:
    assert record.exists()
    info=json.loads(record.read_text())
    for p,backup,expected in zip(files,info['backup_paths'],info['original_sha256']):
        b=Path(backup).read_bytes();assert sha(b)==expected
        p.write_bytes(b)
    info['restored_sha256']=[sha(p.read_bytes()) for p in files]
    record.write_text(json.dumps(info,indent=2)+'\n')
print(record)
