#!/usr/bin/env python3
"""Check source identities, old interval replay and conservative scope labels."""
from pathlib import Path
from hashlib import sha256
import json,statistics,math
ROOT=Path('audit/execution-search')
def main():
 r=json.loads((ROOT/'target-draft-metadata-review.json').read_text());p=json.loads((ROOT/'target-draft-metadata-price.json').read_text());raw=json.loads((ROOT/'target-draft-metadata-price-raw.json').read_text())
 assert len({s['runtime_path'] for s in r['sources']})==15 and len(r['controls'])==34
 for s in r['sources']:assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
 for d in [p,raw]:
  for x in d['inputs']:assert sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
 calls=[c for rank in raw['ranks'] for c in rank['calls']];assert len(raw['ranks'])==16 and len(calls)==176 and all(len(rank['calls'])==11 for rank in raw['ranks']);assert sum(len(rank['unmatched']) for rank in raw['ranks'])==2
 for c in calls:
  assert c['ordinal']==26 and len(c['child_ids'])==3
  assert 0<=c['guarded_application_ms']<=c['application_ms']<=c['free_prepare_ms']<=c['body_wall_ms']+1e-8
  assert math.isclose(c['application_ms']+c['marker_ms'],c['free_prepare_ms'],abs_tol=1e-8)
  for intervals,expected in [(c['guarded_application_free_prepare_us'],c['guarded_application_ms']),(c['application_free_prepare_us'],c['application_ms']),(c['free_prepare_us'],c['free_prepare_ms'])]:assert math.isclose(sum(b-a for a,b in intervals)/1000,expected,abs_tol=1e-8)
 assert math.isclose(statistics.mean(rank['totals_ms']['guarded_application_ms'] for rank in raw['ranks']),15.2234375,abs_tol=1e-8)
 assert math.isclose(p['local_net_scenarios'][2]['conditional_11_frame_ms'],6.51171875,abs_tol=1e-8)
 assert r['original_static_source_bindings_unchanged']==90 and r['original_static_sites_pending_unchanged']==66
 assert r['status']=='待确认' and not r['whole_chain_complete'] and not r['new_primitive_admissions'] and not r['new_framework_replacement_admissions']
 for d in [r,raw]:assert not d['new_NPU_work'] and not d['new_profile'] and not d['service_requests'] and not d['remote_writes']
 artifacts=['target-draft-metadata-primary.json','target-draft-metadata-dependencies.json','target-draft-graph-dependencies.json','target-draft-metadata-review.json','target-draft-metadata-price-raw.json','target-draft-metadata-price.json','target-draft-metadata-source-excerpts.json','target-draft-metadata-frontier.json','target阶段draft_metadata整段交付判断.md']
 out=dict(status='PASS_BOUNDED_SOURCE_AND_ARCHIVED_INTERVAL_INTEGRITY_NOT_NATIVE_ADMISSION',sources=15,original_Python_controls=34,ranks=16,complete_calls=176,unmatched_FT_body_ranges_retained=2,bindings=90,pending=66,whole_chain_complete=False,new_primitive_admissions=0,new_framework_replacement_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],artifacts=[dict(path=str(ROOT/n),sha256=sha256((ROOT/n).read_bytes()).hexdigest()) for n in artifacts],verifier_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
 (ROOT/'target-draft-metadata-integrity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['artifacts','verifier_sha256']}))
if __name__=='__main__':main()
