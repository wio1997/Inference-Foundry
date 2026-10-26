#!/usr/bin/env python3
"""Source-pinned, stage-clock-only extension of Run666 candidate."""
from pathlib import Path
import ast,difflib,hashlib,json
ROOT=Path('/data/wio/Inference_Foundry')
BASE=ROOT/'evidence/20260929_loop081_bound/run666'
OUT=ROOT/'evidence/20260929_loop081_bound/run668'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def one(s,old,new):
    if s.count(old)!=1: raise RuntimeError(f'anchor count {s.count(old)}: {old[:70]}')
    return s.replace(old,new)
def main():
    base=json.loads((BASE/'candidate_manifest.json').read_text())
    cand=OUT/'candidate';diff=OUT/'diff';cand.mkdir(parents=True,exist_ok=False);diff.mkdir(parents=True,exist_ok=False)
    manifest={'status':'source_only_not_installed','sources':{},'helpers':{}}
    for key,row in base['sources'].items():
        path=Path(row['path']); raw=path.read_bytes()
        if sha(raw)!=row['before_sha256']: raise RuntimeError('source drift '+key)
        s=(BASE/'candidate'/f'{key}.py').read_text()
        if sha(s.encode())!=row['candidate_sha256']: raise RuntimeError('base candidate drift '+key)
        if key=='runner':
            s=one(s,'            self._extreme_runtime_started = True\n',
                '            _stage_handoff_entry_ns = time.perf_counter_ns()\n            self._extreme_runtime_started = True\n')
            s=one(s,'            if os.getenv("EXTREME_TARGET_METADATA_GRAPH") == "1":\n',
                '            _stage_graph_begin_ns = time.perf_counter_ns()\n            if os.getenv("EXTREME_TARGET_METADATA_GRAPH") == "1":\n')
            s=one(s,'                _extreme_runtime.target_metadata.capture_graph(_extreme_runtime.state)\n',
                '                _extreme_runtime.target_metadata.capture_graph(_extreme_runtime.state)\n            _stage_graph_end_ns = time.perf_counter_ns()\n')
            s=one(s,'            _stage_graph_begin_ns = time.perf_counter_ns()\n',
                '            _stage_build_done_ns = time.perf_counter_ns()\n            _stage_graph_begin_ns = time.perf_counter_ns()\n')
            s=one(s,'                _extreme_wall_start = time.perf_counter()\n                _cohort_output = FixedCohortServing(',
                '                _stage_runtime_begin_ns = time.perf_counter_ns()\n                _extreme_wall_start = time.perf_counter()\n                _cohort_output = FixedCohortServing(')
            s=one(s,'                torch.npu.synchronize()\n                _extreme_wall_seconds = (\n',
                '                torch.npu.synchronize()\n                _stage_runtime_end_ns = time.perf_counter_ns()\n                _extreme_wall_seconds = (\n')
            s=one(s,'                    "wall_seconds": _extreme_wall_seconds,\n',
                '                    "stage_handoff_entry_ns": _stage_handoff_entry_ns,\n                    "stage_build_done_ns": _stage_build_done_ns,\n                    "stage_graph_begin_ns": _stage_graph_begin_ns,\n                    "stage_graph_end_ns": _stage_graph_end_ns,\n                    "stage_runtime_begin_ns": _stage_runtime_begin_ns,\n                    "stage_runtime_end_ns": _stage_runtime_end_ns,\n                    "wall_seconds": _extreme_wall_seconds,\n')
        ast.parse(s)
        (cand/f'{key}.py').write_text(s)
        d=''.join(difflib.unified_diff(raw.decode().splitlines(True),s.splitlines(True),fromfile=str(path),tofile=str(path)+'.run668'))
        (diff/f'{key}.diff').write_text(d)
        manifest['sources'][key]={'path':str(path),'before_sha256':row['before_sha256'],'candidate_sha256':sha(s.encode()),'diff_sha256':sha(d.encode())}
    (OUT/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'files':len(manifest['sources'])}))
if __name__=='__main__':main()
