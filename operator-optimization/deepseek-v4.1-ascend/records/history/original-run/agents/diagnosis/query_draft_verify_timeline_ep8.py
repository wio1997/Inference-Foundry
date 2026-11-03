#!/usr/bin/env python3
"""Read fixed EP8 windows and four exact task types; preserve absent and clipped cases."""
import argparse,csv,hashlib,json
from collections import Counter
from decimal import Decimal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--task-root',required=True);p.add_argument('--trace-root',required=True);a=p.parse_args()
T=Path(a.task_root);R=Path(a.trace_root)
rank='dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt'
files={'gaps':('inter-forward-gap-analysis-v2/rank_gaps.csv','6c19695c6dfb5cba0bd58b23e21113f1db82f93a20a949715658ea1b81a1c917'),
       'draft':('eager-draft-mainstream-attribution-v2/draft_call_metrics.json','0e6f0cb4d810fdd9f7ec99f7b9a7dbe2123a7db7e5a927115e721b8fd8982c48'),
       'manifest':('evidence/TRACE-EXPORT-SHA256.json','45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139')}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for name,(path,digest) in files.items():assert sha(T/path)==digest,name
with (T/files['gaps'][0]).open(newline='') as f:
 windows=[r for r in csv.DictReader(f) if r['rank']==rank and r['target_forward_ordinal'] in ('6','7')]
assert len(windows)==2 and {r['target_forward_ordinal'] for r in windows}=={'6','7'}
expected_bounds={'6':(1793152960143078904,1793152960186146063),'7':(1793152960223455898,1793152960261768012)}
drafts=json.loads((T/files['draft'][0]).read_text()); br={}
for w in windows:
 n=w['target_forward_ordinal'];start,end=int(w['previous_target_layer39_mc2_end_ns']),int(w['next_target_layer0_start_ns']);assert (start,end)==expected_bounds[n] and start<end
 cand=sorted([r for r in drafts if r['rank']==rank and start<=r['start_ns']<r['end_ns']<=end],key=lambda z:z['start_ns'])
 assert len(cand)==3 and all(i['end_ns']<=j['start_ns'] for i,j in zip(cand,cand[1:]))
 br[n]=[{'start_rel_ns':d['start_ns']-start,'end_rel_ns':d['end_ns']-start,'main_stream':str(d['main_stream_id']),
         'model_id':str(d['model_id']),'source_role':d['source_role'],'block_identity':d['block_identity_within_3']} for d in cand]
manifest=json.loads((T/files['manifest'][0]).read_text());h={r['rank']:r['kernel_csv_sha256'] for r in manifest['exports']}
path=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';assert path.stat().st_size<=50*1024*1024 and sha(path)==h[rank]
wanted={'ArgMaxV2','rejection_greedy_sample_triton','copy_and_expand_dflash_and_dspark_inputs_kernel','_build_dspark_swa_indices_kernel'}
events={n:[] for n in expected_bounds}
with path.open(newline='') as f:
 for z in csv.DictReader(f):
  typ=z['Type']
  if typ not in wanted:continue
  s=int(Decimal(z['Start Time(us)'].strip())*1000);e=s+int(Decimal(z['Duration(us)'].strip())*1000)
  if e<=s:continue
  for n,(start,end) in expected_bounds.items():
   if s>=end or e<=start:continue
   rels=[]
   for b in br[n]:
    bs,be=start+b['start_rel_ns'],start+b['end_rel_ns']
    rels.append('before' if e<=bs else 'after' if s>=be else 'overlaps')
   events[n].append({'type':typ,'stream':z['Stream ID'].strip(),'model':z['Model ID'].strip(),'task_id':z['Task ID'].strip(),
     'start_rel_ns':s-start,'end_rel_ns':e-start,'clipped_start_rel_ns':max(start,s)-start,'clipped_end_rel_ns':min(end,e)-start,
     'complete_in_window':start<=s and e<=end,'relation_to_strict_draft_brackets':rels})
for n,es in events.items():
 assert len(es)<=40,'event cap exceeded';es.sort(key=lambda x:(x['start_rel_ns'],x['end_rel_ns'],x['type']))
out={'scope':'EP8 two prior39-MC2-to-next0 windows; 4 exact Type labels only and three strict brackets each',
     'input_sha256':{k:v[1] for k,v in files.items()},'csv_sha256':h[rank],'rank':rank,'windows':[]}
for n in ('6','7'):
 out['windows'].append({'ordinal':int(n),'start_ns':expected_bounds[n][0],'end_ns':expected_bounds[n][1],
   'strict_draft_brackets':br[n],'types_count':dict(Counter(e['type'] for e in events[n])), 'events':events[n]})
out['limits']=['Windows start after prior layer39 MC2, not previous forward end; ordinal6 prior eager layer mapping is not independently role-validated.',
 'Only four exact type labels, positive durations and selected EP8 CSV; excluded streams/tasks, missing records and unselected outputs prevent completeness claims.',
 'Events/brackets may be unrelated requests; ordinal, timestamp order, overlap, model/stream IDs and names do not prove dependency, source identity or same replay.',
 'Bracket absence is not proof model, attention or other work did not run; physical request/token contents never read.',
 'No NPU operation, profiling, implementation changes or cross-rank clock comparisons.']
print(json.dumps(out,ensure_ascii=False,indent=2))
