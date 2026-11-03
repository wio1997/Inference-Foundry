"""CPU-only synthetic raw files, no NPU or generated response data."""
from pathlib import Path
import json,tempfile
from raw_profile_probe import snapshot,evaluate
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'profile-client-results').mkdir();(p/'diagnostic-traces').mkdir()
 progress={'state':'REQUESTS_COMPLETE_EXPORT_PENDING','profile_stop_http':200,'measure1_counter_deltas':{'request_success_total':8,'prompt_tokens_total':262144,'generation_tokens_total':8192}}
 (p/'profile-client-results/progress.json').write_text(json.dumps(progress))
 for n in range(16):
  r=p/'diagnostic-traces'/('worker'+str(n)+'_ascend_pt');(r/'FRAMEWORK').mkdir(parents=True);(r/'FRAMEWORK/torch.op_range').write_bytes(b'x')
  z=r/'PROF_0'/('device_'+str(n))/'data';z.mkdir(parents=True);(z/'sample.data').write_bytes(b'xx')
 a=snapshot(p);assert not a['issues'] and evaluate([a,a,a])['status']=='RAW_VISIBILITY_STABLE_CLUE'
 x=p/'diagnostic-traces/worker0_ascend_pt/PROF_0/device_0/data/sample.data';x.write_bytes(b'xxx')
 b=snapshot(p);assert evaluate([a,b,b])['status']=='RAW_PROFILE_INCOMPLETE_OR_UNSTABLE'
 x.unlink();c=snapshot(p);assert any('device_data_missing' in x for x in c['issues'])
 progress['profile_stop_http']=500;(p/'profile-client-results/progress.json').write_text(json.dumps(progress));assert 'stop_profile_not_200' in snapshot(p)['issues']
 progress['profile_stop_http']=200;progress['measure1_counter_deltas']['request_success_total']=7;(p/'profile-client-results/progress.json').write_text(json.dumps(progress));assert 'measure1_counters_incomplete' in snapshot(p)['issues']
 progress['state']='FAILED';(p/'profile-client-results/progress.json').write_text(json.dumps(progress));assert 'client_assertions_not_completed' in snapshot(p)['issues']
print(json.dumps({'status':'PASS','scope':'CPU synthetic files; no response file','incomplete_or_unstable':'flagged','three_sample_stability':'PASS'}))
