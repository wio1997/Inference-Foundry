"""Offline audit of original mHC component receipts, not an NPU test."""
from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parents[1]
b=R/'evidence/mHC_interfaces_G371'
a=json.loads((b/'acceptance.json').read_text());assert a['accepted_component'] and not a['full_model_PD_quality_or_performance_accepted']
cases=graphs=replays=0;maxima={}
for rank in range(8):
 row=json.loads((b/'G364_accepted/evidence'/f'rank{rank}.json').read_text());assert row['rank']==rank and row['phase']=='passed'
 assert len(row['cases'])==11
 for c in row['cases']:
  assert c['phase']=='passed' and c['graphs']==1 and c['changed_replays']==3 and len(set(c['hashes']))==3
  assert c['rows'] in ([1,16,96] if c['kind']=='actual_model_mix_stats' else [1,16,96,8192])
  cases+=1;graphs+=c['graphs'];replays+=c['changed_replays'];maxima[c['kind']]=max(maxima.get(c['kind'],0),c['max_abs_error'])
assert (cases,graphs,replays)==(a['cases'],a['graphs'],a['changed_replays'])==(88,88,264)
assert maxima==a['max_abs_error_by_kind']
for name,phase in [('G357_hardware_failure','failed_released'),('G361_invalid_fixture','failed_released'),('G364_accepted','stopped_released')]:
 state=json.loads((b/name/'state.json').read_text());assert state['phase']==phase and state['cleanup']['all8_process_lists_clear']
assert json.loads((b/'fixture_correction.json').read_text())['phase']=='fixture_invalid_not_kernel_verdict'
for p in (R/'code/full_model_candidates/sg_pd_g373').rglob('*.py'):ast.parse(p.read_text())
for e in json.loads((R/'evidence/deployment_provenance.json').read_text()):
 if any(x in e['artifact'] for x in ['mHC_interfaces','sg_pd_g373','full_model_G373']):assert hashlib.sha256((R/e['artifact']).read_bytes()).hexdigest()==e['published_sha256']
print(json.dumps({'offline_mHC_cases':cases,'graphs':graphs,'changed_replays':replays,'original_failures_retained':True,'full_model_PD_accepted':False}))
