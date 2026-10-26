#!/usr/bin/env python3
"""Copy Run580 immutable raw traces and parse only their copies after stop."""
from __future__ import annotations
import hashlib,json,shutil,subprocess
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
RUN=ROOT/'evidence/20260928_loop080_bound/run580'
FINAL=RUN/'final_admission.json'
OUT=RUN/'offline_parse'
CONTAINER='vllm-ascend26-dsv4f-w4a8'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def need(ok,why):
    if not ok:raise ValueError(why)

def main():
    final=json.loads(FINAL.read_text())
    raw=final['raw_profile_provenance']
    need(raw and len(raw)>=8*3,'Run580 final raw profile provenance missing')
    for path,expected in raw.items():need(sha(ROOT/path)==expected,f'raw SHA drift {path}')
    if OUT.exists():need(not (OUT/'manifest.json').exists(),
                         'Run580 offline parse already complete')
    else:OUT.mkdir(parents=True)
    parsed=[]
    for rank in range(8):
        source=RUN/'live/b/bench/profile'/f'rank{rank}'
        target=OUT/'input'/f'rank{rank}'
        if not target.exists():shutil.copytree(source,target)
        copied=[p for p in target.rglob('*') if p.is_file()]
        need(copied and all(sha(p)==raw[str((source/p.relative_to(target)).relative_to(ROOT))]
                            for p in copied),f'rank{rank} raw copy SHA')
        sessions=[p for p in target.iterdir() if p.is_dir() and
                  len(list(p.glob('profiler_info*.json')))==1]
        need(len(sessions)==1,f'rank{rank} profiler session count {len(sessions)}')
        code=('from torch_npu.profiler.profiler import analyse; '
              f'analyse({str(sessions[0])!r}, export_type="text")')
        done=subprocess.run(['docker','exec',CONTAINER,'python3','-c',code],
                            text=True,stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,timeout=180)
        (OUT/f'rank{rank}_analyse.log').write_text(done.stdout)
        need(done.returncode==0,f'rank{rank} offline parse exit {done.returncode}')
        csvs=list(sessions[0].rglob('task_time.csv'))
        need(len(csvs)==1 and csvs[0].stat().st_size>100,
             f'rank{rank} parsed task timeline missing')
        parsed.append({'rank':rank,'raw_file_count':len(copied),
                       'task_time_csv':str(csvs[0].relative_to(ROOT)),
                       'task_time_sha256':sha(csvs[0]),
                       'trace_json':str(next(sessions[0].rglob('trace_view.json')).relative_to(ROOT))})
        print(json.dumps({'rank':rank,'status':'copy_parsed','task_time_bytes':csvs[0].stat().st_size}),flush=True)
    for path,expected in raw.items():need(sha(ROOT/path)==expected,f'postparse raw SHA drift {path}')
    result={'status':'offline_copied_parse_complete','raw_source_immutable':True,
            'final_admission_sha256':sha(FINAL),'parsed':parsed}
    (OUT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'ranks':len(parsed)}))

if __name__=='__main__':main()
