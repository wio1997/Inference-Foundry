#!/usr/bin/env python3
import pathlib,json,hashlib,subprocess,shlex
p=pathlib.Path(__file__).resolve().parents[2];src=p/'agents/diagnosis/query_ep8_15_l14_l39_aicpu_overlap.py';out=p/'agents/diagnosis/ep8-15-l14-l39-aicpu-overlap.json';inp=p/'diagnostic-20261029/target-layer-mc2-16rank.json'
hash0=hashlib.sha256(inp.read_bytes()).hexdigest()
assert hash0=='16d9e17986f59c4caeea1d0b1157f2d839b8344df0fc7c91e1b3105b612448a3'
d=json.loads(inp.read_text());spec=[]
for r in d['ranks']:
 if r['ep'] not in range(8,16):continue
 targets=[]
 for l in r['layers']:
  if l['layer'] not in (14,39) or l['scheduler_digest']!='1b826883b782ba77814d5bd40fcd3e46c759e380069db2e066ff0e9cb4bce89d':continue
  assert l['matched_task_count']==1 and len(l['tasks'])==1
  t=l['tasks'][0];assert t['csv']['Name']=='DispatchFFNCombine'
  targets.append(dict(layer=l['layer'],start_us=t['device_start_us'],duration_us=t['trace_duration_us'],stream=t['csv']['Stream ID'],task=t['csv']['Task ID']))
 assert sorted(x['layer'] for x in targets)==[14,39]
 spec.append(dict(ep=r['ep'],rank=r['rank'],pid=r['pid'],csv_sha=r['input_sha256']['kernel_csv'],targets=targets))
assert sorted(x['ep'] for x in spec)==list(range(8,16))
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';stage='wio-vllm-v41-repro169-stage-20261028'
def call(args,stdin=None,timeout=40):
 r=subprocess.run([ssh,shlex.join(args)],input=stdin,text=True,capture_output=True,timeout=timeout)
 if r.returncode:raise RuntimeError(r.stderr[-2000:])
 return r.stdout
def inspect():
 s=call(['docker','inspect','--format','{{.Id}}|{{.Image}}|{{json .Mounts}}',stage]);cid,img,m=s.strip().split('|',2); m=json.loads(m)
 assert cid=='42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602' and img=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
 assert len([x for x in m if x.get('Destination')=='/work' and x.get('Source')=='/data/tiankuan/wio/main' and x.get('Type')=='bind'])==1
 root=pathlib.PurePosixPath('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime')
 for x in m:
  d=pathlib.PurePosixPath(x['Destination']);assert d==pathlib.PurePosixPath('/work') or not(d==root or d in root.parents or root in d.parents)
 return cid,img,sorted(m,key=lambda x:json.dumps(x,sort_keys=True))
before=inspect(); code='SPEC_JSON='+repr(json.dumps(spec))+'\n'+src.read_text();compile(code,str(src),'exec')
raw=call(['docker','exec','-i',stage,'nice','-n','15','ionice','-c','3','python3','-c',code],timeout=600)
assert before==inspect();j=json.loads(raw);j.update(input_sha256=hash0,script_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),no_new_device_work=True)
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(j,indent=2)+'\n');tmp.replace(out)
for r in j['ranks']:
 print(r['ep'],[(s['layer'],s['raw_rows'],round(s['union_ns']/1e6,3)) for s in r['samples']],flush=True)
