#!/usr/bin/env python3
"""CPU-only read-only bounded query of own historical profiler; no host-device subtraction."""
from pathlib import Path
import hashlib,json,shlex,subprocess
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
stage='wio-vllm-v41-repro169-stage-20261028'
expected_id='42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602'
expected_image='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
code=r'''import pathlib,sqlite3,hashlib,json
root=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
paths=list(root.glob('*ep2_rank*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_2.db')); assert len(paths)==1
p=paths[0];h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
sha=h.hexdigest();assert sha=='2f51a3a608eaf21fbadbb98d54e79032bc23638da76cd80344e9b5c363add7c9'
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.execute('pragma query_only=on');c.row_factory=sqlite3.Row
def name(n):
 r=c.execute('select value from STRING_IDS where id=?',(n,)).fetchone();return r[0] if r else None
def task(start,typ):
 rows=c.execute('select * from TASK where startNs=? and deviceId=2 and globalPid=1113',(start,)).fetchall();assert len(rows)==1
 t=dict(rows[0]);z=c.execute('select name from COMPUTE_TASK_INFO where globalTaskId=?',(t['globalTaskId'],)).fetchone();assert z and name(z[0])==typ;return t
index=task(1793241334237379068,'VllmAscendQuantLightningIndexerV2')
prepare=task(1793241334239479686,'_prepare_indexer_indices_kernel')
att=task(1793241334239631483,'SparseFlashMla')
assert index['streamId']==prepare['streamId']==att['streamId']==47
assert index['endNs']<=prepare['startNs'] and prepare['endNs']<=att['startNs']
a=index['endNs'];b=prepare['startNs']
starts=[dict(r) for r in c.execute('select * from TASK where deviceId=2 and globalPid=1113 and startNs>=? and startNs<? order by startNs',(a,b))]
assert not starts
intersects=[dict(r) for r in c.execute('select * from TASK where deviceId=2 and globalPid=1113 and startNs<? and endNs>? order by startNs',(b,a))]
assert len(intersects)==1
v=intersects[0];assert (v['streamId'],v['taskId'],v['connectionId'])==(index['streamId'],index['taskId'],index['connectionId']) and v['globalTaskId']!=index['globalTaskId']
def cann_by_conn(conn):
 rs=c.execute('select * from CANN_API where connectionId=?',(conn,)).fetchall();assert len(rs)==1
 v=dict(rs[0]);v['decoded_name']=name(v['name']);return v
idxlaunch=cann_by_conn(index['connectionId']);prelaunch=cann_by_conn(prepare['connectionId'])
assert idxlaunch['decoded_name']==prelaunch['decoded_name']=='launch' and idxlaunch['globalTid']==prelaunch['globalTid']
tid=idxlaunch['globalTid']
outer=[dict(z) for z in c.execute('select * from CANN_API where globalTid=? and startNs<=? and endNs>=?',(tid,idxlaunch['startNs'],idxlaunch['endNs']))]
outer=[z for z in outer if name(z['name'])=='aclnnVllmAscendQuantLightningIndexerV2'];assert len(outer)==1;outer=outer[0];outer['decoded_name']=name(outer['name'])
assert outer['startNs']<=idxlaunch['startNs']<=idxlaunch['endNs']<=outer['endNs']<=prelaunch['startNs']
pt=[]
for z in c.execute('select * from PYTORCH_API where globalTid=? and startNs<? and endNs>? order by startNs',(tid,str(prelaunch['endNs']),str(outer['startNs']))):
 z=dict(z);n=name(z['name']);
 if n in ('Dequeue@aclnnVllmAscendQuantLightningIndexerV2','Dequeue@_prepare_indexer_indices_kernel'):
  z['decoded_name']=n;pt.append(z)
assert len(pt)==2 and [z['decoded_name'] for z in pt]==['Dequeue@aclnnVllmAscendQuantLightningIndexerV2','Dequeue@_prepare_indexer_indices_kernel']
assert int(pt[0]['startNs'])<=outer['startNs'] and int(pt[0]['endNs'])>=outer['endNs']
assert int(pt[1]['startNs'])<=prelaunch['startNs'] and int(pt[1]['endNs'])>=prelaunch['endNs']
assert int(pt[0]['endNs'])<=int(pt[1]['startNs'])
print(json.dumps({'db_sha256':sha,'filter':'deviceId=2, globalPid=1113; TASK intervals same DB; host APIs same globalTid',
 'indexer_task':index,'prepare_task':prepare,'attention_task':att,'starts_within_gap_count':len(starts),
 'intersecting_task_rows':intersects,'cann_outer':outer,'cann_indexer_launch':idxlaunch,'cann_prepare_launch':prelaunch,'pytorch_dequeues':pt,
 'device_gap_ns':b-a,'cann_outer_duration_ns':outer['endNs']-outer['startNs'],'cann_outer_after_indexer_launch_ns':outer['endNs']-idxlaunch['endNs'],
 'cann_prepare_launch_after_outer_ns':prelaunch['startNs']-outer['endNs'],
 'pytorch_indexer_dequeue_duration_ns':int(pt[0]['endNs'])-int(pt[0]['startNs']),
 'pytorch_dequeue_separation_ns':int(pt[1]['startNs'])-int(pt[0]['endNs']),
 'limitations':['Only own EP2 historical profiling DB and matching process/device, not full device or other users.','Overlap by timestamp and same thread is not proof of causal blocking or CPU busy.','No subtraction between CANN/PyTorch host and TASK device timestamps; no global idleness inference.','Connection IDs link the two launch records to tasks, not outer call objects or tensor/event identities.']},indent=2))'''
def call(args):
 r=subprocess.run([ssh,shlex.join(args)],capture_output=True,text=True,timeout=90)
 if r.returncode:raise RuntimeError(r.stderr[-1000:])
 return r.stdout
identity=call(['docker','inspect','--format','{{.Id}}|{{.Image}}',stage]).strip()
assert identity==expected_id+'|'+expected_image
raw=call(['docker','exec',stage,'nice','-n','15','ionice','-c','3','python3','-c',code]);d=json.loads(raw)
assert call(['docker','inspect','--format','{{.Id}}|{{.Image}}',stage]).strip()==identity
out=Path(__file__).with_name('ep2-indexer-gap-query.json');out.write_text(json.dumps(d,indent=2)+'\n')
print('wrote',out.name,'sha256',hashlib.sha256(out.read_bytes()).hexdigest())
print({k:d[k] for k in ('device_gap_ns','cann_outer_duration_ns','cann_outer_after_indexer_launch_ns','cann_prepare_launch_after_outer_ns','pytorch_indexer_dequeue_duration_ns','pytorch_dequeue_separation_ns','starts_within_gap_count')})
