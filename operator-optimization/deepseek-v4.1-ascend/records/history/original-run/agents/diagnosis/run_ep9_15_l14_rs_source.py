#!/usr/bin/env python3
import pathlib,json,hashlib,subprocess,shlex
p=pathlib.Path(__file__).resolve().parents[2];src=p/'agents/diagnosis/query_ep9_15_l14_rs_source.py';out=p/'agents/diagnosis/ep9-15-l14-rs-source.json';anchors=p/'agents/diagnosis/ep8-15-l14-l39-aicpu-overlap.json';targets=p/'diagnostic-20261029/target-layer-mc2-16rank.json'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(anchors)=='644597de5a4b73389e5f483f07bc858429227153501f8ec3f298565a1da59bac' and sha(targets)=='16d9e17986f59c4caeea1d0b1157f2d839b8344df0fc7c91e1b3105b612448a3'
a=json.loads(anchors.read_text());t=json.loads(targets.read_text());spec=[]
for r in a['ranks']:
 if r['ep'] not in range(9,16):continue
 rr=[x for x in t['ranks'] if x['ep']==r['ep']];assert len(rr)==1;rr=rr[0]
 s=[x for x in r['samples'] if x['layer']==14];assert len(s)==1
 items=[{k:x[k] for k in ('streamId','taskId','startNs','endNs','connectionId')} for x in s[0]['records'] if x['csv']['Name']=='reduce_scatterAicpuKernel'];assert items
 spec.append({'ep':r['ep'],'pid':rr['pid'],'rank':rr['rank'],'side_sha':rr['input_sha256']['metadata'],'db_sha':r['db_sha256_first_observed'],'items':items})
assert sorted(x['ep'] for x in spec)==list(range(9,16))
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';stage='wio-vllm-v41-repro169-stage-20261028'
def call(args,timeout=40):
 r=subprocess.run([ssh,shlex.join(args)],text=True,capture_output=True,timeout=timeout)
 if r.returncode:raise RuntimeError(r.stderr[-2000:])
 return r.stdout
def inspect():
 s=call(['docker','inspect','--format','{{.Id}}|{{.Image}}|{{json .Mounts}}',stage]);cid,img,m=s.strip().split('|',2);m=json.loads(m)
 assert cid=='42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602' and img=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
 assert len([x for x in m if x.get('Destination')=='/work' and x.get('Source')=='/data/tiankuan/wio/main' and x.get('Type')=='bind'])==1
 root=pathlib.PurePosixPath('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime')
 for x in m:
  d=pathlib.PurePosixPath(x['Destination']);assert d==pathlib.PurePosixPath('/work') or not(d==root or d in root.parents or root in d.parents)
 return cid,img,sorted(m,key=lambda x:json.dumps(x,sort_keys=True))
before=inspect();code='SPEC_JSON='+repr(json.dumps(spec))+'\n'+src.read_text();compile(code,str(src),'exec');raw=call(['docker','exec',stage,'nice','-n','15','ionice','-c','3','python3','-c',code],timeout=600);assert before==inspect()
j=json.loads(raw);j.update(anchor_sha256=sha(anchors),target_sha256=sha(targets),script_sha256=sha(src),wrapper_sha256=sha(pathlib.Path(__file__)),no_new_device_work=True)
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(j,indent=2)+'\n');tmp.replace(out)
for r in j['ranks']:
 print('ep',r['ep'],'items',[(x['task']['taskId'],len(x['hosts']),len(x['comm_ops'])) for x in r['items']],flush=True)
