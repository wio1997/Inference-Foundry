import json,sys,zlib,base64
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
HERE=Path(__file__).resolve().parent
payload=base64.b64encode(zlib.compress((HERE/'creation_profile_probe.py').read_bytes())).decode()
code='PAYLOAD='+repr(payload)+'\n'+r'''import json,subprocess,sys,base64,zlib,importlib.util,re,os,time
from pathlib import Path
c=Path('/data/tiankuan/wio/migration-171-20261102/builtin-cohort-v164-228ad25fcf9b');sys.path.insert(0,str(c))
s=importlib.util.spec_from_file_location('guards',c/'controller.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
root=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/history167-creation');root.mkdir(parents=True,exist_ok=True)
if (root/'HOST-SUBMITTED.json').exists():print(json.dumps({'existing_handle':json.loads((root/'HOST-SUBMITTED.json').read_text())}));raise SystemExit(0)
m=g.inspect(g.MAIN);assert m['State']['Running']
g.idle_metrics(g.request(30075,'/metrics')[1].decode());busy,text=g.busy();g.owns_processes(busy,g.owned(g.MAIN))
mem=g.command(['npu-smi','info','-t','usages','-i','0']);rates=[int(x) for x in re.findall(r'HBM Usage Rate\(%\)\s*:\s*(\d+)',mem)]
assert len(rates)==2 and max(rates)<=92,mem
script=root/'probe.py';script.write_bytes(zlib.decompress(base64.b64decode(PAYLOAD)))
inside='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/analysis/history167-creation'
launcher="import subprocess,signal,os,json;from pathlib import Path;root=Path(%r);log=(root/'device-probe.log').open('w');p=subprocess.Popen(['python3',str(root/'probe.py')],stdout=log,stderr=log,start_new_session=True);(root/'DEVICE-HANDLE.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]})+'\\n');\ntry:rc=p.wait(timeout=75)\nexcept subprocess.TimeoutExpired:\n os.killpg(p.pid,signal.SIGTERM)\n try:rc=p.wait(timeout=5)\n except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait()\n(root/'DEVICE-EXIT.json').write_text(json.dumps({'returncode':rc})+'\\n')"%inside
log=(root/'host.log').open('ab');p=subprocess.Popen(['docker','exec','-e','ASCEND_RT_VISIBLE_DEVICES=0','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',g.MAIN,'python3','-c',launcher],stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
handle={'pid':p.pid,'start_ticks':g.proc_start(p.pid),'root':str(root),'main_started_at_before':m['State']['StartedAt'],'memory_rates_before':rates,'authorization':'Broad user authorization for owned container probes/process operations.','application_tensor_bytes':1160,'runtime_context_overhead_additional':True}
(root/'HOST-SUBMITTED.json').write_text(json.dumps(handle,indent=2)+'\n');print(json.dumps(handle))
'''
result=invoke('target',code)
(HERE/'creation-probe-submission.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
