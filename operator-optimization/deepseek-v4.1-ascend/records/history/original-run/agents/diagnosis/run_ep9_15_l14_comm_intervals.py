#!/usr/bin/env python3
import pathlib,json,hashlib,subprocess,shlex
p=pathlib.Path(__file__).resolve().parents[2];src=p/'agents/diagnosis/query_ep9_15_l14_comm_intervals.py';out=p/'agents/diagnosis/ep9-15-l14-comm-intervals.json';anchors=p/'agents/diagnosis/ep8-15-l14-l39-aicpu-overlap.json';source=p/'agents/diagnosis/ep9-15-l14-rs-source.json';targets=p/'diagnostic-20261029/target-layer-mc2-16rank.json'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(anchors)=='644597de5a4b73389e5f483f07bc858429227153501f8ec3f298565a1da59bac' and sha(source)=='4b6a3b546ed1260cdfee2c2011324c45ad759a7dc15ef866d69b34e60369b851' and sha(targets)=='16d9e17986f59c4caeea1d0b1157f2d839b8344df0fc7c91e1b3105b612448a3'
a=json.loads(anchors.read_text());s=json.loads(source.read_text());t=json.loads(targets.read_text());spec=[]
for r in s['ranks']:
 ep=r['ep'];ar=[x for x in a['ranks'] if x['ep']==ep];tr=[x for x in t['ranks'] if x['ep']==ep];assert len(ar)==len(tr)==1;ar=ar[0];tr=tr[0]
 sample=[x for x in ar['samples'] if x['layer']==14];assert len(sample)==1
 item=[x for x in r['items'] if x['task']['taskId']==3999];assert len(item)==1 and len(item[0]['comm_ops'])==1
 spec.append({'ep':ep,'pid':tr['pid'],'rank':tr['rank'],'db_sha':r['db_sha256'],'anchor':sample[0]['anchor'],'aicpu':item[0]['task'],'op':item[0]['comm_ops'][0]})
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
j=json.loads(raw);j.update(anchor_sha256=sha(anchors),source_sha256=sha(source),target_sha256=sha(targets),script_sha256=sha(src),wrapper_sha256=sha(pathlib.Path(__file__)),no_new_device_work=True)
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(j,indent=2)+'\n');tmp.replace(out)
for r in j['ranks']:print(r['ep'],'op_delay_ms',r['op_start_from_anchor_end_ns']/1e6,'children',r['children_count'],'unique_match',r['task_unique_match_count'],'exact',r['timestamp_exact_match_count'],'overlap',r['overlapping_child_count'],'first_child_delay_ms',r['earliest_child_start_after_anchor_end_ns']/1e6 if r['earliest_child_start_after_anchor_end_ns'] is not None else None,flush=True)
