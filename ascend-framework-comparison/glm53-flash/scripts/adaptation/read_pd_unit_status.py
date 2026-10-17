# SPDX-License-Identifier: Apache-2.0
"""Read-only owned B103 model PD checkpoint; no start/stop/exec/inference."""
import argparse,ast,concurrent.futures,json,pathlib,re,shlex,subprocess

p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);p.add_argument('--logs',action='store_true');p.add_argument('--compact',action='store_true');p.add_argument('--output',type=pathlib.Path);a=p.parse_args();R=a.run
unit=json.loads((R/'DRAFT.json').read_text())['unit'].lower();assert re.fullmatch(r'r\d+',unit)
tree=ast.parse((R/'launcher.py').read_text())
root=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote_root' for t in n.targets))
assert root.startswith('/data/wio/inference-foundry/glm53-flash/runs/20261017-'+unit+'-')
pods={46:'sglang-container-access-46-20261013',47:'sglang-container-access-47-20261013'}
ssh=['ssh','-S','/Users/wio/work/sglang-glm53-pd/ssh-control','-p','60011','-o','BatchMode=yes','-o','ConnectTimeout=15','root@61.241.77.34']
code='''import pathlib,json,subprocess,collections
R=pathlib.Path(ROOT);node=NODE;unit=UNIT;out={'node':node};name='if-glm53-flash-'+unit+'-pd'+str(node)
q=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
if q.returncode:
 out['container_missing']=True
else:
 v=json.loads(q.stdout)[0]
 assert v['Image']=='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263' and v['Config']['Labels']['project_id']=='glm53-flash' and v['Config']['Labels']['run_id']==unit
 out['container_state']={k:v['State'][k] for k in ('Running','ExitCode')}
 if not v['State']['Running']:
  idle=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15);out['actual_stopped_idle_count']=idle.stdout.count('No running processes found') if idle.returncode==0 else None
 out['actual_container_identity']={'container':name,'image':v['Image'],'project_id':v['Config']['Labels']['project_id'],'run_id':v['Config']['Labels']['run_id']}
 for f in ('host-status.json','preflight.json','service-ready.json'):
  if (R/f).exists():
   x=json.loads((R/f).read_text());out[f]={k:x[k] for k in ('passed','service_ready','ready','error','cleanup_error','post_inspect','generation_requests','logical_generation_sequences','decode_stop_permitted') if k in x}
   if f=='preflight.json':out[f].update(imports=len(x.get('imports',{})),bounded_recv=x.get('Flash_native_bounded_recv'),native_admission={'reserved':x.get('resolved',{}).get('num_reserved_decode_tokens'),'capacity':x.get('resolved',{}).get('max_total_tokens'),'input_required_pairs':[[v['input_tokens'],v['native_required_tokens']] for v in x.get('native_empty_pool_admission',[])]})
 out['generation_API_intents']=len(list(R.glob('request-*.once.lock')))
 out['requests']=[]
 for f in sorted(R.glob('request-[0-9]*.json')):
  x=json.loads(f.read_text());out['requests'].append({k:x[k] for k in ('attempted','status','error','sequence') if k in x}|{'sequences':len(x.get('sequences',[]))})
 modes=[];total=0;counts=collections.Counter();devices={}
 observed={'actual_forward_batch','kda_handoff','push_complete','payload_checksum','decode_request_intake','destination_allocated_before_publish','late_write_gate_entered','destination_deferred','prefill_failed_source_retained','decode_admission_blocked_held_mamba','deadline_expired_destination_still_owned','late_write_gate_released','native__send_abort_ack_after','native_note_abort_ack_after','destination_release_after','pd_sdk_close_complete'}
 for f in sorted((R/'pd-events').glob('*.jsonl')):
  total+=f.stat().st_size;assert total<=1024**3
  with f.open('rb') as stream:
   for line in stream:
    if not line.endswith(b'\\n'):continue
    e=json.loads(line);counts[e['kind']]+=1
    if e['kind'] in observed:devices.setdefault(e['kind'],set()).add(e.get('device'))
    if e['kind']=='pd_native_worker_recv_mode':modes.append({k:e[k] for k in ('device','native_timeout_ms','role_switch')})
 out['actual_native_recv_modes']=modes
 out['lifetime_and_close_counts']={k:{'events':counts[k],'unique_devices':sorted(v for v in d if v is not None)} for k,d in devices.items()}
 if LOGS:
  q=subprocess.run(['docker','logs','--tail','30',name],capture_output=True,text=True,timeout=20)
  out['log_tail']=(q.stdout+q.stderr)[-3500:]
print(json.dumps(out))
'''
def one(node):
 source=code.replace('ROOT',repr(root)).replace('NODE',str(node)).replace('UNIT',repr(unit)).replace('LOGS',repr(a.logs))
 command=['kubectl','exec','-i','-n','default',pods[node],'-c','hostctl','--','nsenter','-t1','-m','--','python3','-']
 q=subprocess.run(ssh+[shlex.join(command)],input=source,capture_output=True,text=True,timeout=45)
 return json.loads(q.stdout) if q.returncode==0 else {'read_error':q.stderr[-1500:],'rc':q.returncode}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:out={'project_id':'glm53-flash','unit':unit,'actual_frozen_remote_root':root,'resource_mutations':0,'nodes':dict(zip(pods,ex.map(one,pods)))}
out['progress']=json.loads((R/'progress.json').read_text()) if (R/'progress.json').exists() else None
out['receipt_exists']=(R/'receipt.json').exists()
if a.output:a.output.write_text(json.dumps(out,indent=2)+'\n')
if a.compact:
 brief={'project_id':out['project_id'],'unit':unit,'resource_mutations':0,'nodes':{},'progress':out['progress'],'receipt_exists':out['receipt_exists']}
 for node,v in out['nodes'].items():
  brief['nodes'][node]={k:v[k] for k in ('container_state','actual_stopped_idle_count','host-status.json','generation_API_intents','requests','read_error','container_missing') if k in v}
  brief['nodes'][node]['native_recv_devices']=len(v.get('actual_native_recv_modes',[]))
  brief['nodes'][node]['events']={k:{'events':c['events'],'devices':len(c['unique_devices'])} for k,c in v.get('lifetime_and_close_counts',{}).items()}
 print(json.dumps(brief))
else:print(json.dumps(out))
