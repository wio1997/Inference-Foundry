#!/usr/bin/env python3
"""Locate where first-RS arrival skew reappears between adjacent Target Graphs."""
from __future__ import annotations
import hashlib,json
from decimal import Decimal
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry');BASE=ROOT/'evidence/20260928_loop081_bound'
PATHS={'scope':(BASE/'run611/scope_ledger.json','3c9bb23a28674fae048daad9c32a2803324d5509aa418c41d66c129fe27fd2ff'),
'graph':(BASE/'run611/native_graph_replay.json','9518ee739ece657c8cff467ff476d92e7cc939048945abccc1e9f7e5a009336d'),
'rs':(BASE/'run643/first_rs_arrival.json','0b27bff9e989c5675d9ec18e9557ce2c09619d6e13ed8bede202a3e0892b9b8c')}
OUT=BASE/'run644/precursor_stage_join.json'
def dec(x):return Decimal(str(x))
def main():
 data={}
 for name,(p,digest) in PATHS.items():
  assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,(name,'source drift')
  data[name]=json.loads(p.read_text())
 scope={(x['rank'],x['ordinal_in_profile']):x for x in data['scope']['rows']}
 graph={(x['rank'],x['ordinal_in_profile']):x for x in data['graph']['rows']}
 rs={(x['rank'],x['ordinal']):x for x in data['rs']['rows']}
 allkeys={(rank,i) for rank in range(8) for i in range(3)}
 assert set(scope)==set(graph)==set(rs)==allkeys
 records=[];transitions=[]
 for i in (0,1):
  q=[]
  for rank in range(8):
   prior_graph_end=dec(graph[rank,i]['last_task_end_us'])
   prior_proposer_end=dec(scope[rank,i]['stages']['proposer']['end_us'])
   next_target_host=dec(scope[rank,i+1]['stages']['target']['start_us'])
   next_graph_start=dec(graph[rank,i+1]['first_task_start_us'])
   next_rs_start=dec(rs[rank,i+1]['first_rs_start_us'])
   row={'rank':rank,'transition':f'{i}->{i+1}',
    'prior_graph_end_us':str(prior_graph_end),'prior_proposer_host_end_us':str(prior_proposer_end),
    'next_target_host_start_us':str(next_target_host),'next_graph_start_us':str(next_graph_start),
    'next_first_rs_start_us':str(next_rs_start),
    'prior_graph_end_to_next_graph_start_us':str(next_graph_start-prior_graph_end),
    'prior_graph_end_to_prior_proposer_end_us':str(prior_proposer_end-prior_graph_end),
    'prior_proposer_end_to_next_target_host_start_us':str(next_target_host-prior_proposer_end),
    'next_target_host_start_to_graph_start_us':str(next_graph_start-next_target_host)}
   records.append(row);q.append(row)
  spread=lambda key:str(max(dec(x[key]) for x in q)-min(dec(x[key]) for x in q))
  transitions.append({'transition':f'{i}->{i+1}',
   'prior_graph_end_spread_us':spread('prior_graph_end_us'),
   'prior_proposer_host_end_spread_us':spread('prior_proposer_host_end_us'),
   'next_target_host_start_spread_us':spread('next_target_host_start_us'),
   'next_graph_start_spread_us':spread('next_graph_start_us'),
   'next_first_rs_start_spread_us':spread('next_first_rs_start_us'),
   'last_graph_entry_rank':max(q,key=lambda x:dec(x['next_graph_start_us']))['rank']})
 result={'status':'sameW0_two_transition_precursor_interval_join_pass',
  'scope':'Instrumented Run611 Host/native joined conditional timebase. Intervals are algebraic elapsed scopes, not disjoint service or removable time. Prior Graph end is an exported endpoint, not certified all-device completion. Next Graph start skew is not necessarily avoidable; critical producer-consumer and resource cost remain unknown.',
  'input_sha256':{k:v[1] for k,v in PATHS.items()},'transitions':transitions,'rank_rows':records,
  'framework_only_whole_product_tps_bound':None}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'transitions':transitions}))
if __name__=='__main__':main()
