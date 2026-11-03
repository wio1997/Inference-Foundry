#!/usr/bin/env python3
"""Reproduce bounded L23 communication JSON + child metadata inspection."""
import hashlib,json,pathlib,shlex,subprocess
root=pathlib.Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl';previous=root/'agents/diagnosis/l23-rank-communication.json'
assert sha(source)=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert sha(previous)=='f920ad422a443c28a1f34878c44472ec203618d5598380ef2de01b53edf32633'
rankpaths={}
for line in source.open():
 x=json.loads(line)
 if x.get('kind')=='rank' and x['data']['ep'] in range(8):rankpaths[x['data']['ep']]=x['data']['rank']
spec=[]
for row in json.loads(previous.read_text())['ranks']:
 items=[]
 for i in row['items']:
  assert i['task_match_count']==1 and len(i['comm_ops'])==1
  o=i['comm_ops'][0];items.append({'digest':i['digest'],'opid':o['opId'],'name':o['nameVal'],'connection':o['connectionId'],'start':o['startNs'],'end':o['endNs']})
 spec.append({'ep':row['ep'],'rank':rankpaths[row['ep']],'db_sha':row['db_sha256'],'items':items})
assert len(spec)==8 and all(len(x['items'])==2 for x in spec)
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';stage='wio-vllm-v41-repro169-stage-20261028'
def call(args,timeout=40):
 r=subprocess.run([ssh,shlex.join(args)],text=True,capture_output=True,timeout=timeout)
 if r.returncode:raise RuntimeError(r.stderr[-1500:])
 return r.stdout
def inspect():
 text=call(['docker','inspect','--format','{{.Id}}|{{.Image}}|{{json .Mounts}}',stage]);cid,img,m=text.strip().split('|',2);mounts=json.loads(m)
 assert cid=='42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602' and img=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
 assert len([x for x in mounts if x.get('Destination')=='/work' and x.get('Source')=='/data/tiankuan/wio/main' and x.get('Type')=='bind'])==1
 rootpath=pathlib.PurePosixPath('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime')
 for mount in mounts:
  p=pathlib.PurePosixPath(mount['Destination']);assert p==pathlib.PurePosixPath('/work') or not(p==rootpath or p in rootpath.parents or rootpath in p.parents)
 return cid,img,sorted(mounts,key=lambda x:json.dumps(x,sort_keys=True))
before=inspect();script=(root/'agents/diagnosis/query_l23_comm_json_metrics.py').read_text();code='SPEC='+repr(spec)+'\n'+script;compile(code,'stage_query','exec')
result=call(['docker','exec',stage,'nice','-n','15','ionice','-c','3','python3','-c',code],timeout=120);assert before==inspect()
j=json.loads(result);assert len(j['ranks'])==8
for r in j['ranks']:
 assert len(r['items'])==2
 for x in r['items']:
  assert x['json_entry_count']==1 and x['child_rows']==x['child_unique_global_ids']
  assert abs(float(x['json_start_minus_op_us']))<=0.25 and float(x['json_elapse_minus_op_ms'])==0
print(result,end='')
