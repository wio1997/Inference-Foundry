"""Preserve stopped D raw data, CPU-export all ranks, correlate D2D DMA."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import re
import shutil
import socket
import subprocess
import tarfile
import time

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-ffn-graph-live2-20261018/node33')
gate=json.loads((root/'mtp-ffn-graph-profile-v1/result.json').read_text())
assert gate['completed'] and gate['profile_stopped'] and gate['ranks']==list(range(16))
assert all(x['complete_answer_pass'] for x in gate['rows'])
dirs=[Path(p) for p in gate['new_profile_dirs']]
files=sorted(p for d in dirs for p in d.rglob('*') if p.is_file())
assert all(p.parent==root/'profile' and not p.is_symlink() for p in dirs)
assert all(not p.is_symlink() for p in files)
assert sum(p.stat().st_size for p in files)<4*1024**3 and shutil.disk_usage(root).free>8*1024**3
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False)
snapshot=root/'mtp-ffn-all16-raw-v1';snapshot.mkdir(exist_ok=False)
manifest=[]
for p in files:
    q=snapshot/p.relative_to(root/'profile');q.parent.mkdir(parents=True,exist_ok=True)
    digest=sha(p);shutil.copyfile(p,q);assert sha(q)==digest
    manifest.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest})
(out/'raw-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
shutil.copyfile(root/'mtp-ffn-graph-profile-v1/result.json',out/'profile-result.json')
with tarfile.open(out/'all16-raw.tar.gz','w:gz',compresslevel=3) as tar:
    for p,record in zip(files,manifest):
        tar.add(p,arcname=record['path'],recursive=False)
        assert sha(p)==record['sha256']
image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
argv=['docker','run','--rm','--pull','never','--network','none','--name','glm53-mtp-ffn-all16-export-live2-v1',
      '--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','16','--memory','64g','--pids-limit','4096',
      '--tmpfs','/tmp:rw,size=2g','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1',
      '--device','/dev/davinci_manager','-v',str(snapshot)+':/profiles',
      '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/timeout',image,
      '--signal=TERM','--kill-after=15','600','python3','-c',
      "from torch_npu.profiler.profiler import analyse; analyse('/profiles',max_process_number=4,export_type='text')"]
start=time.monotonic()
with (out/'export-stdout.txt').open('xb') as a,(out/'export-stderr.txt').open('xb') as b:
    proc=subprocess.run(argv,stdout=a,stderr=b,timeout=630)
assert proc.returncode==0,proc.returncode
rows=[]
for directory in sorted(snapshot.glob('*_ascend_pt')):
    rank=int(re.search(r'_rank(\d+)_',directory.name)[1])
    traces=list(directory.rglob('trace_view.json'));assert len(traces)==1,(rank,traces)
    trace=traces[0];obj=json.loads(trace.read_text());events=obj['traceEvents'] if isinstance(obj,dict) else obj
    xevents=[v for v in events if v.get('ph')=='X']
    apis=sorted([v for v in xevents if v.get('name')=='AscendCL@aclrtMemcpyAsync'],key=lambda v:float(v['ts']))
    tasks=defaultdict(list)
    for event in xevents:
        if 'Task Type' in event.get('args',{}) and 'connection_id' in event['args']:
            tasks[event['args']['connection_id']].append(event)
    copies=[]
    for queue in [v for v in xevents if v.get('name')=='Dequeue@acl_memcpy_device_to_device']:
        begin=float(queue['ts']);end=begin+float(queue['dur'])
        matches=[v for v in apis if begin<=float(v['ts'])<=end]
        assert len(matches)==1,(rank,queue,len(matches))
        api=matches[0];device=tasks.get(api['args']['connection_id'],[])
        assert len(device)==1,(rank,api,len(device))
        copies.append({'queue':queue,'api':api,'device':device[0]})
    rank_out=out/('rank%d'%rank);rank_out.mkdir()
    shutil.copyfile(trace,rank_out/'trace_view.json')
    for pattern in ['kernel_details.csv','communication_matrix.json','api_statistic.csv']:
        for p in directory.rglob(pattern):shutil.copyfile(p,rank_out/p.name)
    rows.append({'rank':rank,'trace_sha256':sha(trace),'trace_bytes':trace.stat().st_size,'events':len(events),
                 'D2D_count':len(copies),'D2D_device_task_types':dict(Counter(c['device']['args']['Task Type'] for c in copies)),
                 'all_captured_D2D_SDMA':bool(copies) and all(re.search(r'(?:PCIE_DMA|SDMA)_SQE',c['device']['args']['Task Type']) for c in copies),
                 'rdma_named_X_events':sum('rdma' in e.get('name','').lower() for e in xevents),
                 'D2D':copies})
    del obj,events,xevents,tasks,apis,copies
report={'completed':True,'export_rc':proc.returncode,'export_elapsed_s':time.monotonic()-start,'argv':argv,
        'all16_raw_files':len(manifest),'all16_raw_bytes':sum(x['bytes'] for x in manifest),
        'all16_raw_archive_sha256':sha(out/'all16-raw.tar.gz'),'all16_raw_archive_bytes':(out/'all16-raw.tar.gz').stat().st_size,
        'compute_devices_used':[],'query_device_only':'/dev/davinci_manager','ranks':rows,
        'all16_captured_D2D_SDMA':len(rows)==16 and all(x['all_captured_D2D_SDMA'] for x in rows),
        'scope':'All observed native D2D runtime queues linked via ACL API connection_id to physical device tasks, with ordinary PD8/opt-in MTPFFN graph in same session. Attribution of a subset to private graph input copies requires source/shape/count comparison; no per-byte or graph-wide numerical-equivalence claim, no profiled service timing.'}
(out/'correlation.json').write_text(json.dumps(report,indent=2)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
assert sorted(x['rank'] for x in rows)==list(range(16))
