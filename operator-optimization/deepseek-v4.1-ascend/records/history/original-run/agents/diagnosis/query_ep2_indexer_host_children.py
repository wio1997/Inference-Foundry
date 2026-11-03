#!/usr/bin/env python3
"""Reproduce same-thread host API interval coverage in the owned EP2 profile."""
import hashlib,json,shlex,subprocess
from pathlib import Path
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';stage='wio-vllm-v41-repro169-stage-20261028'
identity='42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602|sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
code=r'''import pathlib,sqlite3,hashlib,json
root=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
paths=list(root.glob('*ep2_rank*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_2.db'));assert len(paths)==1
p=paths[0];h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
assert h.hexdigest()=='2f51a3a608eaf21fbadbb98d54e79032bc23638da76cd80344e9b5c363add7c9'
info=p.parent.parent/'profiler_info_2.json';raw=info.read_bytes();cfg=json.loads(raw)['config']
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.execute('pragma query_only=on');c.row_factory=sqlite3.Row
rs=[dict(r) for r in c.execute('select a.startNs,a.endNs,a.connectionId,s.value name from CANN_API a left join STRING_IDS s on a.name=s.id where a.globalTid=? and a.startNs<? and a.endNs>? order by a.startNs,a.endNs',(4780298607293,1793241334239350840,1793241334237248320))]
outer=[r for r in rs if r['connectionId']==126630];assert len(outer)==1 and outer[0]['name']=='aclnnVllmAscendQuantLightningIndexerV2';outer=outer[0]
child=[r for r in rs if r['connectionId']!=outer['connectionId']];assert len(child)==5
assert all(outer['startNs']<=r['startNs']<r['endNs']<=outer['endNs'] for r in child)
assert [(r['connectionId'],r['name']) for r in child]==[(126631,'aclrtGetStreamAttribute'),(126632,'aclrtGetHardwareSyncAddr'),(126633,'launch'),(126634,'aclrtLaunchKernelWithHostArgs'),(126635,'aclrtGetStreamAttribute')]
launch=next(r for r in child if r['connectionId']==126633);nested=next(r for r in child if r['connectionId']==126634)
assert launch['startNs']<=nested['startNs']<nested['endNs']<=launch['endNs']
intervals=sorted((r['startNs'],r['endNs']) for r in child);merged=[]
for a,b in intervals:
 if not merged or a>merged[-1][1]:merged.append([a,b])
 else:merged[-1][1]=max(merged[-1][1],b)
covered=sum(b-a for a,b in merged);duration=outer['endNs']-outer['startNs'];tail=outer['endNs']-max(r['endNs'] for r in child)
assert (duration,covered,tail)==(2102520,15450,2071020)
tables=[r[0] for r in c.execute("select name from sqlite_master where type='table' order by name")]
print(json.dumps({'db_sha256':h.hexdigest(),'info_sha256':hashlib.sha256(raw).hexdigest(),'config':cfg,'tables':tables,'outer':outer,'children':child,'merged_intervals_ns':merged,'outer_duration_ns':duration,'children_union_ns':covered,'uncovered_by_child_cann_ns':duration-covered,'last_child_to_outer_end_ns':tail,'limitations':['Same globalTid in CANN_API, overlap/containment only; no host/device timestamp subtraction.','Union measures known instrumented CANN API coverage, not time spent in CPU execution, OS wait, device wait, or attribution.','No table identified as OS scheduling trace in this database, but other non-inspected profiler resources might exist.','Profiled single historical sample only; no unprofiled performance attribution.']},indent=2))'''
def call(a):
 r=subprocess.run([ssh,shlex.join(a)],capture_output=True,text=True,timeout=90)
 if r.returncode:raise RuntimeError(r.stderr[-1000:])
 return r.stdout
assert call(['docker','inspect','--format','{{.Id}}|{{.Image}}',stage]).strip()==identity
raw=call(['docker','exec',stage,'nice','-n','15','ionice','-c','3','python3','-c',code]);data=json.loads(raw)
assert call(['docker','inspect','--format','{{.Id}}|{{.Image}}',stage]).strip()==identity
p=Path(__file__).with_name('ep2-indexer-host-children.json');p.write_text(json.dumps(data,indent=2)+'\n');print('json_sha256',hashlib.sha256(p.read_bytes()).hexdigest(),'info_sha256',data['info_sha256']);print({k:data[k] for k in ('outer_duration_ns','children_union_ns','uncovered_by_child_cann_ns','last_child_to_outer_end_ns')})
