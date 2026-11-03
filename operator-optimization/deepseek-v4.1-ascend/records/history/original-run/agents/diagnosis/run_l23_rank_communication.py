#!/usr/bin/env python3
"""Bounded read-only extraction of existing L23 selected HCCL candidates."""
import hashlib, json, pathlib, shlex, subprocess
root=pathlib.Path(__file__).resolve().parents[2]
source=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
expected='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
spec=[]
for line in source.open():
 x=json.loads(line)
 if x.get('kind')!='rank' or x['data']['ep'] not in range(8):continue
 d=x['data']; tasks=[]
 for l in d['layers']:
  if l['layer']==23 and l['scheduler_digest'][:8] in ('7ddf7ae9','db852bff'):
   a=[t for g in l['operator_groups'] if g['operator_name']=='reduce_scatterAicpuKernel' for t in g['tasks']]
   assert len(a)==1
   t=a[0]; tasks.append({'digest':l['scheduler_digest'][:8], 'start_us':t['device_start_us'],'stream':t['csv']['Stream ID'],'task':t['csv']['Task ID']})
 assert len(tasks)==2
 spec.append({'ep':d['ep'],'pid':d['pid'],'rank':d['rank'],'tasks':tasks})
assert len(spec)==8 and {x['ep'] for x in spec}==set(range(8))
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';stage='wio-vllm-v41-repro169-stage-20261028'
def call(args,timeout=40):
 r=subprocess.run([ssh,shlex.join(args)],text=True,capture_output=True,timeout=timeout)
 if r.returncode:raise RuntimeError(r.stderr[-1500:])
 return r.stdout
def inspect():
 text=call(['docker','inspect','--format','{{.Id}}|{{.Image}}|{{json .Mounts}}',stage]); cid,img,m=text.strip().split('|',2); mounts=json.loads(m)
 assert cid=='42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602' and img=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
 assert len([x for x in mounts if x.get('Destination')=='/work' and x.get('Source')=='/data/tiankuan/wio/main' and x.get('Type')=='bind'])==1
 rootpath=pathlib.PurePosixPath('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime')
 for mount in mounts:
  p=pathlib.PurePosixPath(mount['Destination']); assert p==pathlib.PurePosixPath('/work') or not(p==rootpath or p in rootpath.parents or rootpath in p.parents)
 return cid,img,sorted(mounts,key=lambda x:json.dumps(x,sort_keys=True))
before=inspect(); script=(root/'agents/diagnosis/query_l23_rank_communication.py').read_text();code='SPEC='+repr(spec)+'\n'+script;compile(code,'inside_stage','exec')
result=call(['docker','exec',stage,'nice','-n','15','ionice','-c','3','python3','-c',code],timeout=180);assert before==inspect()
j=json.loads(result)
assert len(j['ranks'])==8
for rank in j['ranks']:
 assert not rank.get('error') and rank['dtype11_enum']==[{'id':11,'name':'BFP16'}]
 assert len(rank['items'])==2
 for item in rank['items']:
  assert item['task_match_count']==1 and len(item['hosts'])==1 and item['hosts'][0]['name']=='HcclReduceScatter'
  assert len(item['comm_ops'])==1
  op=item['comm_ops'][0];m=op['group_metadata']; assert op['same_opid_count']==1 and op['rankSize']==8 and op['dataType']==11
  assert m['group_name']=='tp' and m['global_ranks']==list(range(8)) and m['global_ranks'][m['group_rank']]==rank['rankmap'][0]['rankId']
  assert (op['opId'],op['count']) == ((351,3046400) if item['digest']=='7ddf7ae9' else (209,5237760))
print(result,end='')
