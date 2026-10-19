#!/usr/bin/env python3
"""Archive own terminalDecode log and exact capture/dispatch/KV-limit source; keep P running."""
import ast,datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R069-decode-bounds';O.mkdir(parents=True,exist_ok=False)
tree=ast.parse((R/'scripts/collect_L039_tiling_details.py').read_text());JUMP=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='JUMP' for t in n.targets))
HOST=r'''
import hashlib,json,pathlib,subprocess,sys
j=json.load(sys.stdin);base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36'
p=subprocess.run(['docker','inspect','--format','{{json .}}',j['container']],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr
x=json.loads(p.stdout);assert x['Id']==j['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']=='L039-R067' and not x['State']['Running']
root=base/'L039-R069-decode-diagnostics-46';root.mkdir(mode=0o700,exist_ok=False);raw=root/'model-docker.log'
with raw.open('xb') as f:p=subprocess.run(['docker','logs','--timestamps',j['container']],stdout=f,stderr=subprocess.STDOUT,timeout=40)
data=raw.read_bytes();lines=data.decode(errors='replace').splitlines();v={'node':46,'container_id':x['Id'],'state':x['State'],'raw_log_path':str(raw),'raw_log_sha256':hashlib.sha256(data).hexdigest(),'raw_log_bytes':len(data),'stage_lines':[],'first_OOM_contexts':[],'stopped_services':False,'inference_request_submitted':False}
for line in lines:
 if any(k in line for k in ['Capture sizes','capture_bs','capture bs','bs:','Capturing','KV Cache is allocated','max_total_num_tokens','num_max_dispatch_tokens','Load weight end']):
  if len(v['stage_lines'])<120:v['stage_lines'].append(line[:2500])
indexes=[i for i,line in enumerate(lines) if 'NPU out of memory' in line or 'Tried to allocate' in line]
for i in indexes[:4]:v['first_OOM_contexts'].append('\n'.join(lines[max(0,i-90):min(len(lines),i+75)])[:26000])
print(json.dumps(v))
'''
IMAGE=r'''
import hashlib,json,pathlib,sys
root=pathlib.Path('/sgl-workspace/sglang');v={'sources':{},'Torch_imported':'torch' in sys.modules,'NPU_device_nodes_absent':not any(pathlib.Path('/dev').glob('davinci*'))}
names=['python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py','python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py','python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py','python/sglang/srt/layers/moe/token_dispatcher/deepep.py','python/sglang/srt/environ.py','python/sglang/srt/server_args.py']
for p in (root/'python/sglang/srt/arg_groups').rglob('*.py'):
 text=p.read_text()
 if 'max_total_tokens' in text or 'max-total-tokens' in text:names.append(str(p.relative_to(root)))
for name in names:
 p=root/name
 if not p.is_file():continue
 data=p.read_bytes();assert len(data)<1024*1024;v['sources'][name]={'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'source':data.decode()}
assert not v['Torch_imported'] and v['NPU_device_nodes_absent'];print(json.dumps(v))
'''
h=json.loads((R/'specs/L039-R069-targets.json').read_text());payload={'host':HOST,'image':IMAGE,'targets':h['targets'],'leases':h['resource_leases'],'store':h['store']}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=220)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only_except_own_log_archive':True,'NPU_computation':False,'services_stopped':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
