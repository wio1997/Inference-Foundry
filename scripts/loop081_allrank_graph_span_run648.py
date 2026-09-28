#!/usr/bin/env python3
"""All-rank Target Graph entry/finish decomposition for Run611 same W0."""
from __future__ import annotations
import hashlib,json
from decimal import Decimal
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry');BASE=ROOT/'evidence/20260928_loop081_bound'
G=BASE/'run611/native_graph_replay.json';R=BASE/'run643/first_rs_arrival.json'
OUT=BASE/'run648/allrank_graph_span.json'
G_SHA='9518ee739ece657c8cff467ff476d92e7cc939048945abccc1e9f7e5a009336d'
R_SHA='0b27bff9e989c5675d9ec18e9557ce2c09619d6e13ed8bede202a3e0892b9b8c'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def d(v):return Decimal(str(v))
def main():
 assert sha(G)==G_SHA and sha(R)==R_SHA
 g=json.loads(G.read_text());r=json.loads(R.read_text())
 assert len(g['rows'])==24 and len(r['rows'])==24
 rr={(x['rank'],x['ordinal']):x for x in r['rows']};gg={(x['rank'],x['ordinal_in_profile']):x for x in g['rows']}
 assert set(rr)==set(gg)=={(rank,i) for rank in range(8) for i in range(3)}
 rows=[]
 for i in range(3):
  q=[gg[rank,i] for rank in range(8)]
  starts=[d(x['first_task_start_us']) for x in q];ends=[d(x['last_task_end_us']) for x in q]
  earliest_rank=min(range(8),key=lambda rank:starts[rank]);latest_rank=max(range(8),key=lambda rank:starts[rank])
  earliest=min(starts);latest=max(starts);finish=max(ends)
  entry_skew=latest-earliest;global_span=finish-earliest;late_rank_span=ends[latest_rank]-latest
  result={'ordinal':i,'earliest_graph_entry_rank':earliest_rank,'latest_graph_entry_rank':latest_rank,
    'graph_entry_spread_us':str(entry_skew),'graph_exported_finish_spread_us':str(max(ends)-min(ends)),
    'allrank_earliest_entry_to_last_exported_end_us':str(global_span),
    'latest_entry_rank_local_graph_span_us':str(late_rank_span),
    'allrank_span_minus_latest_rank_span_us':str(global_span-late_rank_span),
    'first_rs_arrival_spread_us':r['occurrences'][i]['first_rs_arrival_spread_us'],
    'first_rs_finish_spread_us':r['occurrences'][i]['first_rs_finish_spread_us']}
  assert abs((global_span-late_rank_span)-entry_skew)<Decimal('100')
  assert latest_rank==r['occurrences'][i]['latest_arrival_rank']==7
  rows.append(result)
 out={'status':'sameW0_all8_graph_entry_completion_decomposition_pass',
  'scope':'Run611 Level0 instrumented and conditionally common cross-rank timebase. All-rank first-entry to last-exported-end span equals entry skew plus latest-rank local span, within exported end dispersion; this is algebra, not an optimization. Delaying earlier ranks cannot advance latest completion. To gain, legally advance latest-rank ready/entry or use independent work while waiting with real contention. Exported end is not all-device completion or Product wall.',
  'graph_replay_sha256':sha(G),'first_rs_sha256':sha(R),'occurrences':rows,
  'framework_only_whole_product_tps_bound':None}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'status':out['status'],'occurrences':rows}))
if __name__=='__main__':main()
