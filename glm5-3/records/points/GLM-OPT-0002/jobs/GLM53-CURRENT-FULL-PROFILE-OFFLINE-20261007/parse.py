"""Official CPU-only offline parser of already captured all16 FULL traces."""
from pathlib import Path
import json,hashlib,shlex,subprocess,time,sys,os
J=Path(__file__).resolve().parent
R=Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0273/profiles_167')
RUN=J.parent.parent/'runs/GLM-RUN-0276'
def write(name,value):
    p=J/name;assert not p.exists(),name;p.write_text(json.dumps(value,indent=2)+'\n')
def remote(argv,timeout=800):
    return subprocess.run(['ssh','-o','BatchMode=yes','root@172.16.10.167',shlex.join(argv)],capture_output=True,timeout=timeout)
def snapshot():
    code="""from pathlib import Path
import hashlib,json
r=Path(%r);folders=list(r.glob('*_ascend_pt'));assert len(folders)==16
rows=[]
for d in folders:
 for p in d.rglob('*'):
  if p.is_file() and any(x in p.parts for x in ('FRAMEWORK','data')):
   rows.append(dict(path=str(p),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(rows))
"""%str(R)
    p=remote(['/usr/bin/python3','-c',code],90);assert p.returncode==0,p.stderr[-2000:];return json.loads(p.stdout)
def main():
    assert json.loads((RUN/'state.json').read_text())['status']=='failed'
    assert json.loads((RUN/'terminal_reconciliation.json').read_text())['all16_health_idle']
    write('state.json',dict(status='running',pid=os.getpid(),started_ns=time.time_ns(),NPU_requests=0,only_existing_raw=True))
    before=snapshot();write('raw_before_167.json',before)
    code='from torch_npu.profiler.profiler import analyse;analyse('+repr(str(R))+',max_process_number=4)'
    start=time.monotonic();p=remote(['docker','exec','-e','ASCEND_RT_VISIBLE_DEVICES=','glm52-single','/usr/local/python3.12.13/bin/python3','-c',code],800)
    (J/'parser.stdout').write_bytes(p.stdout);(J/'parser.stderr').write_bytes(p.stderr);write('parser.json',dict(exit_code=p.returncode,wall_s=time.monotonic()-start,NPU_visible_devices='',model_imports=0,inference_requests=0))
    assert p.returncode==0,p.stderr[-3000:]
    after=snapshot();write('raw_after_167.json',after);assert {z['path']:z for z in before}=={z['path']:z for z in after},'raw capture changed'
    code="""from pathlib import Path
import hashlib,json,re
r=Path(%r);rows=[]
for p in r.rglob('trace_view.json'):
 match=re.search(r'_tp(\d+)_',str(p));rows.append(dict(path=str(p),rank=int(match.group(1)),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(rows))
"""%str(R)
    p=remote(['/usr/bin/python3','-c',code],90);assert p.returncode==0,p.stderr[-2000:];traces=json.loads(p.stdout);write('trace_inputs.json',traces)
    assert len(traces)==16 and {z['rank'] for z in traces}==set(range(16)), 'parser acknowledgment not actual all16 export'
    write('result.json',dict(status='completed',raw_files=len(before),raw_bytes=sum(z['size'] for z in before),preexisting_raw_unchanged=True,exported_ranks=sorted(z['rank'] for z in traces),trace_bytes=sum(z['size'] for z in traces),trace_root=str(R),NPU_requests=0,model_reload=0,service_changes=0,attribution_pending=True))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not (J/'failure.json').exists():write('failure.json',dict(error=repr(e)))
        raise
