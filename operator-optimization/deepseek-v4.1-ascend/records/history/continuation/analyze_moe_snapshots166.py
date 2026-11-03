"""CPU-only decode and conditional same-ordinal cross-rank conservation review."""
import collections,csv,gzip,hashlib,json,os,pathlib,statistics,time
import torch
RUN=os.environ['WIO_RUN_ID'];HOST='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/'+RUN
ROOT=pathlib.Path('/work/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030')/RUN
assert json.loads((ROOT/'controller-finished.json').read_text())['state']=='CAPTURED_AND_RESTORED'
assert json.loads((ROOT/'restore-result.json').read_text())['status']=='PASS_ORIGINAL_BASELINE_RESTORED'
OUT=ROOT/'value-review';OUT.mkdir(exist_ok=False);index=json.loads((ROOT/'tensor-index.json').read_text());groups=collections.defaultdict(list);errors=[];records=[];pointer_layers=collections.defaultdict(set)
def translate(s):
 assert s.startswith(HOST+'/'),s
 return ROOT/s[len(HOST)+1:]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with gzip.open(OUT/'decoded-records.jsonl.gz','wt') as decoded:
 for n,item in enumerate(index):
  try:
   p=translate(item['tensor']);mp=translate(item['metadata']);assert sha(p)==item['tensor_sha256'] and sha(mp)==item['metadata_sha256'];meta=json.loads(mp.read_text());values=torch.load(p,map_location='cpu',weights_only=True)
   assert isinstance(values,list) and len(values)==len(meta['payload_names']);payload=dict(zip(meta['payload_names'],values));c=payload['expert_counts'];ids=payload['topk_ids'];ordinal=payload['observer_ordinal']
   assert c.dtype==torch.int32 and c.ndim==1 and c.numel()==meta['E_local'];assert ids.shape==(meta['M'],meta['K']) and ids.dtype in (torch.int32,torch.int64);assert ordinal.dtype==torch.int64 and ordinal.numel()==1
   counts=[int(x) for x in c.tolist()];epoch=int(ordinal.item());assert epoch>0 and all(x>=0 for x in counts)
   if meta['mask_is_none']:mask=[True]*meta['M'];mask_domain='None argument => all input rows, subject to primitive contract'
   else:
    m=payload['x_active_mask'];assert m.dtype in (torch.bool,torch.int8) and m.numel()==meta['M'],('mask row domain unresolved',m.shape,meta['M']);mask=[bool(x) for x in m.flatten().tolist()];mask_domain='captured same-call mask, one value per input row'
   rows=ids.tolist();hist=collections.Counter(int(e) for active,row in zip(mask,rows) if active for e in row);assert all(0<=e<meta['E_global'] for e in hist),('active routing ID outside expert domain',hist)
   members=tuple(meta.get('ep_group_ranks',[]));namespace=('members',members) if len(members)==meta['ep_world_size'] else ('unresolved-group-name',meta['group'])
   key=(meta['role'],meta['layer_name'],namespace,meta['E_global'],meta['K'],epoch)
   rec={'index':n,'tensor':item['tensor'],'metadata':item['metadata'],'role':meta['role'],'layer':meta['layer_name'],'ep_rank':meta['ep_rank'],'ep_world':meta['ep_world_size'],'group_ranks':list(members),'group_namespace':namespace,'ordinal':epoch,'M':meta['M'],'K':meta['K'],'E_local':meta['E_local'],'E_global':meta['E_global'],'valid_rows':sum(mask),'sender_assignment_count':sum(hist.values()),'sender_expert_histogram':dict(hist),'receiver_expert_counts':counts,'receiver_count_sum':sum(counts),'capacity_max_output_size':meta['capacity_max_output_size'],'receiver_count_sum_above_capacity_argument':sum(counts)>meta['capacity_max_output_size'],'counts_ptr':meta['counts_data_ptr'],'use_mega_moe':meta['use_mega_moe'],'enable_fused_mc2':meta['enable_fused_mc2'],'capturing_entry':meta['capturing'],'mask_domain':mask_domain,'placement_strategy':meta['placement_strategy'],'redundant_experts':meta['global_redundant_experts'],'group_binding_complete':namespace[0]=='members'}
   decoded.write(json.dumps(rec,separators=(',',':'))+'\n');records.append(rec);groups[key].append(rec);pointer_layers[(rec['role'],rec['ep_rank'],rec['counts_ptr'])].add(rec['layer'])
  except Exception as e:errors.append({'index':n,'tensor':item['tensor'],'error':repr(e)})
cohorts=[];conflicts=[]
for key,items in groups.items():
 rankmap=collections.defaultdict(list)
 for rec in items:rankmap[rec['ep_rank']].append(rec)
 world=items[0]['ep_world'];complete=set(rankmap)==set(range(world)) and items[0]['group_binding_complete'];ambiguous=False;chosen=[]
 for rank,dups in rankmap.items():
  fingerprints={(tuple(r['receiver_expert_counts']),tuple(sorted(r['sender_expert_histogram'].items())),r['M'],r['K']) for r in dups}
  if len(fingerprints)>1:ambiguous=True;conflicts.append({'key':str(key),'rank':rank,'record_indices':[r['index'] for r in dups]})
  chosen.append(dups[0])
 row={'role':key[0],'layer':key[1],'ordinal':key[-1],'ep_world':world,'ranks_present':sorted(rankmap),'complete_membership_and_rank_coverage':complete,'ambiguous_duplicate_records':ambiguous,'records':len(items),'sender_assignments':None,'receiver_assignments':None,'total_conservation':None,'linear_placement_domain':None,'per_expert_histogram_match':None}
 if complete and not ambiguous:
  hist=collections.Counter()
  for rec in chosen:hist.update(rec['sender_expert_histogram'])
  sent=sum(hist.values());received=sum(r['receiver_count_sum'] for r in chosen);linear=all(str(r['placement_strategy']).lower()=='linear' and r['redundant_experts']==0 and r['E_global']==r['E_local']*world for r in chosen)
  expected={r['ep_rank']:[hist[r['ep_rank']*r['E_local']+i] for i in range(r['E_local'])] for r in chosen}
  match=all(r['receiver_expert_counts']==expected[r['ep_rank']] for r in chosen) if linear else None
  row.update(sender_assignments=sent,receiver_assignments=received,total_conservation=sent==received,linear_placement_domain=linear,per_expert_histogram_match=match,receiver_sum_min=min(r['receiver_count_sum'] for r in chosen),receiver_sum_max=max(r['receiver_count_sum'] for r in chosen),receiver_sum_mean=received/world,max_count_in_one_expert=max(max(r['receiver_expert_counts'],default=0) for r in chosen),sum_valid_sender_rows=sum(r['valid_rows'] for r in chosen),M_histogram=dict(collections.Counter(r['M'] for r in chosen)),mismatching_ranks=[r['ep_rank'] for r in chosen if linear and r['receiver_expert_counts']!=expected[r['ep_rank']]])
  if match is False:row['expected_counts_by_rank']=expected;row['actual_counts_by_rank']={r['ep_rank']:r['receiver_expert_counts'] for r in chosen}
 cohorts.append(row)
jsondump=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
jsondump(OUT/'cohorts.json',cohorts);jsondump(OUT/'decode-errors.json',errors);jsondump(OUT/'duplicate-conflicts.json',conflicts)
complete=[x for x in cohorts if x['complete_membership_and_rank_coverage'] and not x['ambiguous_duplicate_records']];good=[x for x in complete if x['total_conservation'] is True and x['per_expert_histogram_match'] is True];bad=[x for x in complete if x['total_conservation'] is False or x['per_expert_histogram_match'] is False]
by_layer=[]
for role,layer in sorted({(r['role'],r['layer']) for r in records}):
 rs=[r for r in records if (r['role'],r['layer'])==(role,layer)];cs=[c for c in complete if (c['role'],c['layer'])==(role,layer)];gs=[c for c in good if (c['role'],c['layer'])==(role,layer)];by_layer.append({'role':role,'layer':layer,'snapshots':len(rs),'ranks':sorted({r['ep_rank'] for r in rs}),'complete_cohorts':len(cs),'exact_per_expert_matches':len(gs),'M_values':sorted({r['M'] for r in rs}),'K_values':sorted({r['K'] for r in rs}),'capacity_arguments':sorted({r['capacity_max_output_size'] for r in rs}),'use_mega_moe_values':sorted({r['use_mega_moe'] for r in rs})})
jsondump(OUT/'layer-coverage.json',by_layer)
summary={'status':'PASS_OBSERVED_COHORTS_CONSERVATION_ONLY' if good and not errors and not bad and not conflicts else 'PARTIAL_OR_INCONCLUSIVE_VALUE_REVIEW','scope':'Measured same-call snapshots in observer-perturbed fixed166 case; per-layer observer ordinal is not scheduler/request identity','raw_tensor_files':len(index),'decoded_snapshots':len(records),'decode_errors':len(errors),'duplicate_conflicts':len(conflicts),'cohorts_total':len(cohorts),'cohorts_complete':len(complete),'cohorts_exact_per_expert_matches':len(good),'cohorts_mismatch':len(bad),'cohorts_incomplete_or_ambiguous':len(cohorts)-len(complete),'layer_surfaces':len(by_layer),'layers_by_role':dict(collections.Counter(x['role'] for x in by_layer)),'all_snapshot_ep_ranks':sorted({r['ep_rank'] for r in records}),'use_mega_moe_values':sorted({r['use_mega_moe'] for r in records}),'capacity_arguments':sorted({r['capacity_max_output_size'] for r in records}),'shared_count_pointer_layers_max':max((len(v) for v in pointer_layers.values()),default=0),'matched_cohort_examples':good[:8],'mismatch_examples':bad[:3],'ordinal_alignment_limit':'Cross-rank matched same ordinal is validated by exact routing histogram when matched; no claim that it identifies a CPU scheduler step or every iteration','timing_limit':'No per-call timing bound to these snapshots; do not combine current counts with old169 kernel times as same-call measurements','whole_modules_closed':0,'overall_goal_complete':False,'review_finished_unix':time.time()};jsondump(OUT/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False))
