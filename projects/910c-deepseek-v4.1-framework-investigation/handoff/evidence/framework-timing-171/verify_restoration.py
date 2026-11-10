from pathlib import Path
import json,subprocess
P=Path(__file__).resolve().parent
root=json.loads((P/'launch.json').read_text())['remote']
source='''import hashlib,json,subprocess,sys,urllib.request,re
from pathlib import Path
root=Path(sys.argv[1]);receipt=json.load(open(root/'receipt.json'));assert receipt['status']=='CAPTURED_RESTORED'
base=receipt['original'];info=json.loads(subprocess.check_output(['docker','inspect',base]))[0];assert info['State']['Running']
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open('http://127.0.0.1:30075/health',timeout=10) as r:health=r.status
with opener.open('http://127.0.0.1:30075/metrics',timeout=10) as r:metrics=r.read().decode()
active=[float(l.rsplit(' ',1)[1]) for l in metrics.splitlines() if re.match(r'vllm:num_requests_(running|waiting)\\{',l)]
assert len(active)>=4 and all(v==0 for v in active),'service busy; no smoke request issued'
request=urllib.request.Request('http://127.0.0.1:30075/v1/completions',data=json.dumps({'model':'dsv41','prompt':'Hi','max_tokens':8,'temperature':0,'ignore_eos':True}).encode(),headers={'Content-Type':'application/json'})
with opener.open(request,timeout=180) as r:result=json.load(r)
adapter=json.load(open(root/'adapter.json'));paths={k:v['source_path'] for k,v in adapter['modules'].items()}
program='import json,hashlib,sys;from pathlib import Path;print(json.dumps({k:hashlib.sha256(Path(p).read_bytes()).hexdigest() for k,p in json.loads(sys.argv[1]).items()}))'
hashes=json.loads(subprocess.check_output(['docker','exec',base,'python3','-B','-I','-S','-c',program,json.dumps(paths)]))
assert all(hashes[k]==v['sha256'] for k,v in adapter['modules'].items())
env=info['Config']['Env'];assert not any(x.startswith(('FRAMEWORK_TIMING_ROOT=','FRAMEWORK_TIMING_ADAPTER=','VLLM_CUSTOM_SCOPES_FOR_PROFILING=1')) for x in env)
traces=sorted((root/'traces').rglob('trace_view.json'));spans=sorted((root/'timing').glob('spans-*.jsonl'))
inventory=[]
for p in traces+[q.with_name('kernel_details.csv') for q in traces]+spans:
 b=p.read_bytes();inventory.append({'path':str(p.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
summary={'health':health,'original_image':info['Image'],'original_id':info['Id'],'usage':result['usage'],'original_source_hashes':hashes,'all_adapter_sources_unchanged':True,'original_config_has_no_diagnostic_flags':True,'diagnostic_running':json.loads(subprocess.check_output(['docker','inspect',receipt['diagnostic']]))[0]['State']['Running'],'worker_traces':len(traces),'span_files':len(spans)}
(root/'restored-smoke.json').write_text(json.dumps(summary,indent=2));(root/'raw-inventory.json').write_text(json.dumps(inventory,indent=2));print(json.dumps(summary))
'''
r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',source,root],capture_output=True,text=True,timeout=240)
(P/'restore-check-output.txt').write_text(r.stdout+r.stderr);r.check_returncode();print(r.stdout)
