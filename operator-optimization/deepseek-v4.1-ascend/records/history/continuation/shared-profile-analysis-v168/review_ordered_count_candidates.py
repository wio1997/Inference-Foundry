"""Candidate same-layer ordinal matching is a test,not an unproved global ID."""
import collections,json,statistics,time
from pathlib import Path
ROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint168');OUT=ROOT/'derived'
files=sorted(OUT.glob('*-rows.jsonl'));by=collections.defaultdict(lambda:collections.defaultdict(list));allrows=[]
for f in files:
 for line in f.read_text().splitlines():
  r=json.loads(line);allrows.append(r);by[(r['role'],r['layer_name'])][r['ep_rank']].append(r)
q=lambda xs:{'n':len(xs),'min':min(xs),'median':statistics.median(xs),'max':max(xs)} if xs else {'n':0}
complete=[];incomplete=[];groups=[];failures=[]
for key,ranks in by.items():
 for rank,rs in ranks.items():rs.sort(key=lambda r:r['history_epoch'])
 lengths={r:len(xs) for r,xs in ranks.items()}
 if set(ranks)!=set(range(16)) or len(set(lengths.values()))!=1:incomplete.append({'role':key[0],'layer':key[1],'lengths':lengths});continue
 complete.append({'role':key[0],'layer':key[1],'per_rank_records':next(iter(lengths.values()))})
 for n in range(next(iter(lengths.values()))):
  rows=[ranks[r][n] for r in range(16)];K={r['K'] for r in rows};assert len(K)==1;K=next(iter(K));recv=sum(r['receiver_rows'] for r in rows);valid=sum(r['valid_sender_rows'] for r in rows);expected=K*valid
  g={'role':key[0],'layer':key[1],'candidate_layer_ordinal':n,'all_M3':all(r['M']<=3 for r in rows),'M_values':sorted({r['M'] for r in rows}),'valid_global_rows_under_matching':valid,'receiver_assignment_rows_under_matching':recv,'expected':expected,'difference':recv-expected,'empty_experts_under_matching':sum(r['empty_local_experts'] for r in rows),'max_receiver_rank_rows_under_matching':max(r['receiver_rows'] for r in rows),'rank_max_over_mean_under_matching':max(r['receiver_rows'] for r in rows)/(recv/16) if recv else None}
  groups.append(g)
  if recv!=expected:failures.append(g)
small=[r for r in allrows if r['role']=='target' and r['M']<=3];large=[r for r in allrows if r['role']=='target' and r['M']>3]
selected=[]
for p in OUT.glob('*-selected-ids.json'):
 selected.extend(json.loads(p.read_text()))
ids={'records':len(selected),'all_physical_rows_distinct':sum(r['all_physical_rows_distinct'] for r in selected),'bad_physical_rows_but_no_valid_sender_rows':sum((not r['all_physical_rows_distinct'] or not r['all_physical_ids_in_range']) and r['valid_sender_rows']==0 for r in selected),'bad_physical_rows_with_positive_valid_count':sum((not r['all_physical_rows_distinct'] or not r['all_physical_ids_in_range']) and r['valid_sender_rows']>0 for r in selected)}
summary={'status':'CANDIDATE_ORDERED_LABEL_MATCH_CONSERVATION_TEST','generated_unix':time.time(),'workers_files':len(files),'actual_rows':len(allrows),'complete_layer_sequences':complete,'incomplete_layer_sequences':incomplete,'candidate_groups':len(groups),'conservation_failures':len(failures),'failure_samples':failures[:20],
 'actual_target_small_row_local_facts':{'records':len(small),'receiver_rows':q([r['receiver_rows'] for r in small]),'receiver_rows_above288':sum(r['receiver_rows']>288 for r in small),'valid_sender_rows':q([r['valid_sender_rows'] for r in small]),'empty_local_experts':q([r['empty_local_experts'] for r in small])},
 'actual_target_prefill_local_facts':{'records':len(large),'M_values':sorted({r['M'] for r in large}),'receiver_rows':q([r['receiver_rows'] for r in large])},
 'conditional_joint_M3_candidates':{'groups':sum(g['role']=='target' and g['all_M3'] for g in groups),'empty_experts':q([g['empty_experts_under_matching'] for g in groups if g['role']=='target' and g['all_M3']]),'rank_max_over_mean':q([g['rank_max_over_mean_under_matching'] for g in groups if g['role']=='target' and g['all_M3'] and g['rank_max_over_mean_under_matching'] is not None])},'selected_ID_physical_row_checks':ids,
 'identity_status':'NOT_ESTABLISHED_BY_CONSERVATION_ALONE. Matching is conditional on the separately recorded common-start,communicator-instance,complete-native-coverage and per-group-order premises.','overall_complete':False}
(OUT/'ORDERED-COUNT-CANDIDATE-REVIEW.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'ordered-count-candidate-groups.jsonl').write_text(''.join(json.dumps(g,separators=(',',':'))+'\n' for g in groups));print(json.dumps({k:v for k,v in summary.items() if k not in ['complete_layer_sequences','failure_samples']}))
