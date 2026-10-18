#!/usr/bin/env python3
"""CPU-export all16 already-stopped P profiles and correlate every D2H task."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
    root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32');gate=json.loads((root/'profile-request-v1/profile-request.json').read_text());assert gate['completed'] and gate['profile_stopped']
    selected={}
    for p in (root/'profile').glob('*_ascend_pt'):
        m=re.fullmatch(r'dp0_pp0_tp(\d+)_dcp0_ep\d+_rank(\d+)_.*_ascend_pt',p.name)
        if m and m[1]==m[2]:selected[int(m[1])]=p
    assert set(selected)==set(range(16)) and shutil.disk_usage(root).free>6*1024**3
    snapshot=root/'profile-all-raw-v1';snapshot.mkdir(exist_ok=False);out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False);manifest=[]
    for rank,folder in sorted(selected.items()):
        for p in sorted(folder.rglob('*')):
            if not p.is_file():continue
            assert not p.is_symlink();digest=sha(p);q=snapshot/folder.name/p.relative_to(folder);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);assert sha(q)==digest==sha(p)
            manifest.append({'rank':rank,'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest})
    (out/'raw-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    cmd=['docker','run','--rm','--pull','never','--network','none','--name','glm53-seed-p-all16-export-v1','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','16','--memory','64g','--pids-limit','4096','--tmpfs','/tmp:rw,size=2g','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','--device','/dev/davinci_manager','-v',str(snapshot)+':/profiles','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/timeout',image,'--signal=TERM','--kill-after=15','600','python3','-c',"from torch_npu.profiler.profiler import analyse; analyse('/profiles', max_process_number=2, export_type='text')"]
    start=time.monotonic()
    with (out/'stdout.txt').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:proc=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=630)
    rows=[]
    for rank,folder in sorted(selected.items()):
        traces=list((snapshot/folder.name).rglob('trace_view.json'));assert len(traces)==1,(rank,traces)
        p=traces[0];obj=json.loads(p.read_text());events=obj.get('traceEvents',[]) if isinstance(obj,dict) else obj;copies=[]
        for dequeue in (v for v in events if v.get('name')=='Dequeue@acl_memcpy_device_to_host'):
            t=float(dequeue['ts']);api=[v for v in events if v.get('ph')=='X' and v.get('name')=='AscendCL@aclrtMemcpyAsync' and t<=float(v['ts'])<=t+dequeue['dur']];assert len(api)==1
            conn=api[0]['args']['connection_id'];tasks=[v for v in events if v.get('ph')=='X' and v.get('args',{}).get('connection_id')==conn and 'Task Type' in v.get('args',{})];assert len(tasks)==1
            copies.append({'dequeue':dequeue,'api':api[0],'task':tasks[0]})
        dst=out/f'rank{rank}'/'trace_view.json';dst.parent.mkdir();shutil.copyfile(p,dst)
        rows.append({'rank':rank,'trace_sha256':sha(dst),'trace_bytes':dst.stat().st_size,'D2H':copies,'all_D2H_PCIE_DMA':bool(copies) and all(v['task']['args']['Task Type']=='PCIE_DMA_SQE' for v in copies),'rdma_named_X_events':sum(1 for v in events if v.get('ph')=='X' and 'rdma' in v.get('name','').lower())})
    report={'rc':proc.returncode,'duration_s':time.monotonic()-start,'argv':cmd,'raw_files':len(manifest),'raw_bytes':sum(v['bytes'] for v in manifest),'ranks':rows,'all16_D2H_PCIE_DMA':len(rows)==16 and all(v['all_D2H_PCIE_DMA'] and not v['rdma_named_X_events'] for v in rows),'scope':'All captured producer D2H calls correlated through ACL connection IDs to device tasks. Includes the opt-in existing-MTP draft export in a local P request. Not full PD timing; baseline node/inter-node KV paths unchanged and separately evidenced.'}
    (out/'correlation.json').write_text(json.dumps(report,indent=2)+'\n');print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END');assert proc.returncode==0 and report['all16_D2H_PCIE_DMA']


if __name__=='__main__':main()
