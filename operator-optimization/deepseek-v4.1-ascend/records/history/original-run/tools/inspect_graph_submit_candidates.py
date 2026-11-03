"""Search bounded CANN graph-execute candidates for captured layer0 calls.

Association by rank, candidate thread and temporal proximity is a hypothesis,
not a model-handle or operation-ID link. No timing-derived upper bound results.
"""
from pathlib import Path
from decimal import Decimal
from collections import defaultdict,Counter
import bisect,csv,json,re,sqlite3,statistics
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'graph-submit-candidate-analysis';O.mkdir(exist_ok=False)
targets=defaultdict(list)
for x in csv.DictReader((T/'mc2-preceding-analysis/rank_layer0_timeline.csv').open()):
 x['layer_start_ns']=int(x['layer_start_ns']);x['target_forward_ordinal']=int(x['target_forward_ordinal']);targets[x['rank']].append(x)
details=[];rank_info=[]
for rank,calls in sorted(targets.items()):
 match=re.search(r'_rank\d+_(\d+)_',rank);assert match
 pid=int(match.group(1));tid=(pid<<32)|pid
 db=next((ROOT/rank/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'))
 c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
 q='''SELECT a.startNs,a.endNs,a.connectionId,a.globalTid FROM CANN_API a JOIN STRING_IDS s ON a.name=s.id
      WHERE a.globalTid=? AND s.value='aclmdlRIExecuteAsync' ORDER BY a.startNs'''
 executes=[{'start_ns':int(a),'end_ns':int(b),'connection_id':int(cid) if cid is not None else None,'global_tid':int(tid0)} for a,b,cid,tid0 in c.execute(q,(tid,))]
 tables=[r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'") if any(k in r[0].upper() for k in ['MODEL','GRAPH'])]
 c.close();starts=[e['start_ns'] for e in executes]
 matched=Counter()
 for x in calls:
  end=x['layer_start_ns'];left=bisect.bisect_left(starts,end-2_000_000);right=bisect.bisect_right(starts,end)
  candidates=executes[left:right];matched[str(len(candidates))]+=1
  rec={'rank':rank,'dp':x['dp'],'target_forward_ordinal':x['target_forward_ordinal'],
       'layer_start_ns':end,'captured_run_in_model':int(x['captured_run_in_model']),
       'interior_runs_1_2_3':x['interior_runs_1_2_3']=='True','guessed_thread_id':tid,
       'candidate_count_in_2ms':len(candidates),'candidate_submit_start_ns':None,'candidate_submit_end_ns':None,
       'candidate_connection_id':None,'submit_start_to_layer_start_us':None,'submit_end_to_layer_start_us':None,
       'proven_replay_connection':False}
  if len(candidates)==1:
   e=candidates[0];rec.update(candidate_submit_start_ns=e['start_ns'],candidate_submit_end_ns=e['end_ns'],candidate_connection_id=e['connection_id'],
    submit_start_to_layer_start_us=(end-e['start_ns'])/1000,submit_end_to_layer_start_us=(end-e['end_ns'])/1000)
  details.append(rec)
 rank_info.append({'rank':rank,'filename_pid':pid,'guessed_main_tid':tid,'graph_execute_api_count':len(executes),
                   'capture_layer0_candidate_counts':dict(matched),'model_graph_tables':tables})
with (O/'rank_layer0_submit_candidates.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
groups=defaultdict(list)
for d in details:groups[d['target_forward_ordinal']].append(d)
paired=[]
for o,rs in sorted(groups.items()):
 if len(rs)!=16 or any(r['candidate_count_in_2ms']!=1 for r in rs):continue
 offset=min(r['layer_start_ns'] for r in rs);rec={'ordinal':o,'all_interior_runs_1_2_3':all(r['interior_runs_1_2_3'] for r in rs)}
 for dp in ['dp0','dp1']:
  xx=[r for r in rs if r['dp']==dp];assert len(xx)==8
  rec[dp+'_mean_layer0_offset_us']=statistics.mean((r['layer_start_ns']-offset)/1000 for r in xx)
  rec[dp+'_mean_candidate_submit_offset_us']=statistics.mean((r['candidate_submit_start_ns']-offset)/1000 for r in xx)
  rec[dp+'_mean_submit_to_layer_us']=statistics.mean(r['submit_start_to_layer_start_us'] for r in xx)
 paired.append(rec)
doc={'status':'TEMPORAL_CANDIDATES_ONLY','rank_count':len(rank_info),'captured_layer0_calls':len(details),
     'candidate_count_distribution':dict(Counter(r['candidate_count_in_2ms'] for r in details)),
     'rank_info':rank_info,'candidate_paired_forwards':paired,'proof_of_model_handle_or_request_identity':False,
     'limitations':['Candidate graph execute API on guessed main thread, within2ms before captured layer entry; no model-handle or replay connection is captured here.',
       'GlobalTid encoding is a heuristic pending profiler source confirmation.',
       'Cross-rank ordinal alignment remains a hypothesis; CPU/device timestamps lack independent clock-error bounds.',
       'Measured submit-to-layer delay includes preceding model root tasks and is not a universal lower bound.'],
     'no_device_work':True}
assert len(json.dumps(doc).encode())<32768
(O/'summary.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(doc,indent=2))
