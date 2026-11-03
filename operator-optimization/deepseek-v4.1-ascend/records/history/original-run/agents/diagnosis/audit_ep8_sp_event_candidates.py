#!/usr/bin/env python3
"""Offline exact-ID EP8 event candidates; no assumptions about event handle identity."""
import pathlib,sqlite3,json
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA automatic_index=OFF')
ids=[88654,88655,88658,88670,88683,88684,88685]
marks=','.join('?' for _ in ids)
api=[dict(r) for r in c.execute(f'SELECT a.startNs,a.endNs,a.type,a.globalTid,a.connectionId,s.value name FROM CANN_API a LEFT JOIN STRING_IDS s ON a.name=s.id WHERE a.connectionId IN ({marks}) ORDER BY a.startNs',ids)]
tasks=[dict(r) for r in c.execute(f'SELECT startNs,endNs,deviceId,connectionId,globalPid,taskType,streamId,taskId FROM TASK WHERE connectionId IN ({marks}) ORDER BY startNs',ids)]
pyids=[126595,126596,126597,126598,126599,126600,174588]; marks2=','.join('?' for _ in pyids)
py=[dict(r) for r in c.execute(f'SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,p.type,s.value name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON p.name=s.id WHERE p.connectionId IN ({marks2}) ORDER BY CAST(p.startNs AS INTEGER)',pyids)]
links=[dict(r) for r in c.execute('SELECT id,connectionId FROM CONNECTION_IDS WHERE id=? ORDER BY connectionId',(174588,))]
def get_task(id):
 t=[t for t in tasks if t['connectionId']==id]
 assert len(t)==1
 return t[0]
r1=get_task(88654);w1=get_task(88655);r2=get_task(88683);r3=get_task(88684);w2=get_task(88685)
out={'api':api,'pytorch':py,'tasks':tasks,'selected_host_collective_connection_links':links,'candidate_device_end_differences_ns':{'stream39_wait_end_minus_stream47_record_end':w1['endNs']-r1['endNs'],'stream47_wait_end_minus_stream39_record1_end':w2['endNs']-r2['endNs'],'stream47_wait_end_minus_stream39_record2_end':w2['endNs']-r3['endNs']},'limits':'Exact connectionId joins on a single existing profile, timestamps are displayed/exported coordinates, not proof of event handle equality, data dependence, causal pairing, or calibrated cross-track timing.'}
print(json.dumps(out,indent=2))
