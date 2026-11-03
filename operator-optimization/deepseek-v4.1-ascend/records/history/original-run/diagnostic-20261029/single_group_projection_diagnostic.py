"""One guarded, isolated operator diagnostic; never patches or restarts the service."""
import base64
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
HARNESS=r'''
import json,time,statistics,torch,torch_npu
out={'scope':'isolated synthetic BF16 shape diagnostic, not model/graph throughput','shape_x':[24,1,4096],'shape_w':[1,4096,1024],'device':'ASCEND_RT_VISIBLE_DEVICES=0','target_source_modified':False}
try:
 torch.npu.set_device(0)
 limiter=getattr(torch.npu,'set_per_process_memory_fraction',None)
 if limiter is None:raise RuntimeError('Required per-process allocator cap unavailable; stop')
 limiter(0.02)
 if not hasattr(torch_npu,'npu_transpose_batchmatmul'):raise RuntimeError('Native API missing; stop')
 torch.manual_seed(1729)
 x=(torch.randn(24,1,4096,dtype=torch.float32)*0.1).to(torch.bfloat16).to('npu:0')
 w=(torch.randn(1,4096,1024,dtype=torch.float32)*0.1).to(torch.bfloat16).to('npu:0')
 def native():
  return torch_npu.npu_transpose_batchmatmul(x,w,bias=None,scale=None,perm_x1=(1,0,2),perm_x2=(0,1,2),perm_y=(1,0,2),batch_split_factor=1)
 def plain():return torch.matmul(x.view(24,4096),w.view(4096,1024))
 with torch.inference_mode():
  for _ in range(4):native();plain()
  torch.npu.synchronize()
  a=native().view(24,1024).cpu();b=plain().cpu()
  out['output_check']={'bitwise_equal':torch.equal(a,b),'max_abs_difference':float((a.float()-b.float()).abs().max()),'max_abs_reference':float(a.float().abs().max()),'both_finite':bool(torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all())}
  samples={'native_transpose_batchmatmul':[],'plain_2d_matmul':[]}
  funcs={'native_transpose_batchmatmul':native,'plain_2d_matmul':plain}
  for i in range(16):
   order=list(funcs) if i%2==0 else list(reversed(funcs))
   for name in order:
    start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
    start.record();result=funcs[name]();end.record();end.synchronize()
    samples[name].append(float(start.elapsed_time(end))*1000)
  out['event_interval_us']=samples
  out['median_event_us']={k:statistics.median(v) for k,v in samples.items()}
  out['max_memory_allocated_bytes']=int(torch.npu.max_memory_allocated())
  out['max_memory_reserved_bytes']=int(torch.npu.max_memory_reserved())
  out['torch_version']=torch.__version__;out['torch_npu_version']=getattr(torch_npu,'__version__',None)
  out['status']='MEASURED_ISOLATED_EAGER_INTERVALS'
  out['limits']='Device event intervals include eager issue spacing; same synthetic inputs/weights, warm caches, no graph/profile. Does not validate historical tensor path, model accuracy or end-to-end speedup.'
except Exception as e:
 out['status']='ERROR_STOPPED_NO_RETRY';out['error']=type(e).__name__+': '+str(e)
print(json.dumps(out))
'''
remote=r'''
import base64,json,subprocess,re
TARGET='wio-vllm-v41-repro169-20261028';STAGE='wio-vllm-v41-repro169-stage-20261028'
def call(args,timeout=15):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr[-1200:])
 return p.stdout
def idle():
 code="import urllib.request,json; o=urllib.request.build_opener(urllib.request.ProxyHandler({})); r=o.open('http://127.0.0.1:8172/health',timeout=3); assert r.status==200; s=o.open('http://127.0.0.1:8172/metrics',timeout=3).read().decode(); a=[float(l.rsplit(' ',1)[-1]) for l in s.splitlines() if not l.startswith('#') and ('vllm:num_requests_running{' in l or 'vllm:num_requests_waiting{' in l)]; assert len(a)==4; print(json.dumps({'running_waiting_values':a,'idle':all(v==0 for v in a)}))"
 return json.loads(call(['docker','exec',TARGET,'python3','-c',code]))
before=idle();assert before['idle'],'Owned service has requests; no diagnostic started'
smi=call(['docker','exec',STAGE,'npu-smi','info'])
top=call(['docker','top',TARGET,'-eo','pid,comm'])
owned={int(m.group(1)) for m in re.finditer(r'^\s*(\d+)\s+\S+',top,re.M)}
processes=[int(m.group(1)) for m in re.finditer(r'^\|\s*0\s+0\s*\|\s*(\d+)\s*\|',smi,re.M)]
assert processes and set(processes)<=owned,'Selected chip has unknown/non-owned processes'
memline=next(l for l in smi.splitlines() if re.match(r'^\|\s*0\s+0\s*\|',l) and '0000:' in l)
used,total=map(int,re.findall(r'(\d+)\s*/\s*(\d+)',memline)[-1]);assert total-used>=4096,'Less than 4 GiB free; no diagnostic started'
harness=base64.b64decode(HARNESS_B64).decode()
p=subprocess.run(['docker','exec','-e','ASCEND_RT_VISIBLE_DEVICES=0','-e','TMPDIR=/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029',STAGE,'timeout','--kill-after=3s','40s','python3','-c',harness],capture_output=True,text=True,timeout=50)
after=idle()
out={'guard_before':before,'selected_chip_owned_processes':processes,'selected_chip_free_MiB':total-used,'returncode':p.returncode,'stderr_tail':p.stderr[-3000:],'guard_after':after,'service_restart_or_config_change':False,'diagnostic_stdout':p.stdout}
print(json.dumps(out))
'''
remote='HARNESS_B64='+repr(base64.b64encode(HARNESS.encode()).decode())+'\n'+remote
p=subprocess.run([SSH,shlex.join(['python3','-c',remote])],capture_output=True,text=True,timeout=90)
previous_path=ROOT/'diagnostic-20261029/single-group-projection-diagnostic.json'
previous=json.loads(previous_path.read_text()) if previous_path.exists() else None
out={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'ssh_returncode':p.returncode}
if previous and previous.get('status')=='UNPARSED_DIAGNOSTIC_STOPPED':
 out['previous_attempts']=previous.get('previous_attempts',[])
 if previous.get('previous_attempt'):out['previous_attempts'].append(previous['previous_attempt'])
 out['previous_attempts'].append({'checked_utc':previous.get('checked_utc'),'launcher_sha256':previous.get('launcher_sha256'),'returncode':previous.get('returncode'),'error_tail':previous.get('stderr_tail'),'status':previous.get('status'),'correction':'Preserve raw stdout and find the JSON measurement record independently of trailing runtime messages.'})
if p.returncode:
 out['status']='NOT_COMPLETED_STOPPED';out['error_tail']=p.stderr[-2000:];out['stdout_tail']=p.stdout[-2000:]
else:
 out.update(json.loads(p.stdout))
 out['diagnostic_stdout_tail']=out.get('diagnostic_stdout','')[-12000:]
 for line in reversed(out.get('diagnostic_stdout','').splitlines()):
  try:record=json.loads(line)
  except Exception:continue
  if isinstance(record,dict) and record.get('scope','').startswith('isolated synthetic BF16'):
   out['measurement']=record;out['status']=record.get('status');break
 else:out['status']='UNPARSED_DIAGNOSTIC_STOPPED'
(ROOT/'diagnostic-20261029/single-group-projection-diagnostic.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
