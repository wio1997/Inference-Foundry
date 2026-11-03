"""Measured counterexamples and conditional expert-placement lower bounds."""
import collections,json,math,statistics,time
from pathlib import Path
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint168')
O=P.parent/'joint169';W=35389440

def stats(xs):return {'n':len(xs),'min':min(xs),'median':statistics.median(xs),'max':max(xs),'mean':statistics.mean(xs)} if xs else {'n':0}
# Per-layer source ordinals are verified unique;not invented by nearest timestamps.
ordinal={};source_sequences=collections.defaultdict(list)
for p in (P/'derived').glob('*-rows.jsonl'):
 for line in p.read_text().splitlines():
  r=json.loads(line);source_sequences[(r['pid'],r['role'],r['layer_name'])].append(r)
for rs in source_sequences.values():
 rs.sort(key=lambda r:r['history_epoch']);assert len({r['history_epoch'] for r in rs})==len(rs)
 for i,r in enumerate(rs):ordinal[(r['pid'],r['bank_slot'],r['history_epoch'])]=i
rows=[];groups=collections.defaultdict(list);native_by_source={};conditions=collections.Counter();examples=collections.defaultdict(list)
for p in O.glob('*-MC2-JOINS.jsonl'):
 for line in p.read_text().splitlines():
  r=json.loads(line);rows.append(r);native_by_source[(r['pid'],r['bank_slot'],r['history_epoch'])]=r
  vs=r['valid_sender_rows'];recv=r['receiver_rows']
  tags=[]
  if vs==0 and recv>0:tags.append('ZERO_LOCAL_SENDERS_POSITIVE_RECEIVERS')
  if recv==0 and vs>0:tags.append('ZERO_LOCAL_RECEIVERS_POSITIVE_SENDERS')
  if recv==0 and vs==0:tags.append('ZERO_LOCAL_BOTH')
  for tag in tags:
   conditions[tag]+=1
   if len(examples[tag])<4:examples[tag].append({k:r[k] for k in ['pid','ep_rank','layer_name','bank_slot','history_epoch','valid_sender_rows','receiver_rows','native_duration_ns','native_TASK_ROW']})
assert len(rows)==145280
for (pid,role,layer),seq in source_sequences.items():
 if role!='target':continue
 for i,r in enumerate(seq):
  j=native_by_source.get((r['pid'],r['bank_slot'],r['history_epoch']))
  if j:r=dict(r,native_duration_ns=j['native_duration_ns'])
  groups[(role,layer,i)].append(r)
ratios=[];certs=[];replica_need={};tot_current_rows=tot_lower_rows=tot_avg_rows=tot_current_active=tot_lower_active=0
for key,rs in groups.items():
 assert len(rs)==16 and {r['ep_rank'] for r in rs}==set(range(16)) and all(r['M']<=8192 and r['E_local']==24 for r in rs)
 rs.sort(key=lambda r:r['ep_rank']);loads=[v for r in rs for v in r['receiver_counts']];n=sum(loads);A=sum(v>0 for v in loads);maxrows=max(r['receiver_rows'] for r in rs);maxactive=max(24-r['empty_local_experts'] for r in rs);avg_lb=(n+15)//16
 sortedloads=sorted(loads,reverse=True);prefix_lb=max((((i+16)//16)*v for i,v in enumerate(sortedloads)),default=0);row_lb=max(avg_lb,prefix_lb);act_lb=(A+15)//16
 assert row_lb<=maxrows and act_lb<=maxactive and n==sum(r['valid_sender_rows'] for r in rs)*6
 extra=0
 if avg_lb:
  copy_lb=[max(1,(v+avg_lb-1)//avg_lb) for v in loads];extra=sum(copy_lb)-384
  need=replica_need.setdefault(key[1],[1]*384)
  for i,v in enumerate(copy_lb):need[i]=max(need[i],v)
 cert={'layer':key[1],'candidate_layer_ordinal':key[2],'M_values':sorted({r['M'] for r in rs}),'all_M3':all(r['M']<=3 for r in rs),'candidate_identity':'CONDITIONAL_COMMON_START_COMPLETE_ORDERED_SEMANTIC_EP_GROUP','assignment_rows':n,'active_experts':A,'current_max_rank_rows':maxrows,'arithmetic_average_row_LB':avg_lb,'whole_expert_unsplit_prefix_LB':row_lb,'hottest_expert_rows':max(loads),'current_max_active_experts':maxactive,'active_expert_LB':act_lb,
  'whole_expert_row_proxy_improvement_cap':maxrows/row_lb if row_lb else None,'naive_average_only_proxy_cap':maxrows/avg_lb if avg_lb else None,'active_expert_proxy_improvement_cap':maxactive/act_lb if act_lb else None,'extra_full_weight_pairs_LB_for_ideal_row_splitting':extra,
  'rank_native_R_us':stats([r['native_duration_ns']/1000 for r in rs if 'native_duration_ns' in r])}
 certs.append(cert);tot_current_rows+=maxrows;tot_lower_rows+=row_lb;tot_avg_rows+=avg_lb;tot_current_active+=maxactive;tot_lower_active+=act_lb
static=[]
for layer,need in sorted(replica_need.items()):static.append({'layer':layer,'minimum_static_full_weight_pair_copies':sum(need),'extra_pairs_above_original384':sum(need)-384,'extra_int8_weight_bytes_LB':W*(sum(need)-384)})
zero=[r for r in rows if r['receiver_rows']==0];first=[r for r in rows if r['target_layer_observation_order']==0];others=[r for r in rows if r['target_layer_observation_order']!=0]
q_total=sum(r['conditional_target_direct_MAC'] for r in rows);time_total=sum(r['native_duration_ns'] for r in rows)
summary={'status':'ALL_TARGET_PREFILL_AND_SMALL_ROW_PLACEMENT_LOWER_BOUNDS','generated_unix':time.time(),'native_joined_records':len(rows),'source_per_layer_epoch_duplicates':0,'actual_guard_cases':dict(conditions),'actual_guard_examples':dict(examples),
 'measured_native_R_us':stats([r['native_duration_ns']/1000 for r in rows]),'zero_receiver_native_R_us':stats([r['native_duration_ns']/1000 for r in zero]),'zero_receiver_over1ms':sum(r['native_duration_ns']>=1000000 for r in zero),'first_layer_R_us':stats([r['native_duration_ns']/1000 for r in first]),'other_layers_R_us':stats([r['native_duration_ns']/1000 for r in others]),'over1ms_total':sum(r['native_duration_ns']>=1000000 for r in rows),'over1ms_first_layer':sum(r['native_duration_ns']>=1000000 for r in first),
 'direct_target_MAC_in_source_domain':q_total,'cumulative_native_R_work_ns':time_total,'MAC_per_second_of_cumulative_native_kernel_time':q_total/(time_total/1e9),
 'all_target_source_records':sum(len(seq) for (pid,role,layer),seq in source_sequences.items() if role=='target'),'all_target_direct_MAC_in_source_domain':sum(r['conditional_target_direct_MAC'] for (pid,role,layer),seq in source_sequences.items() if role=='target' for r in seq),'conditional_group_count':len(certs),'conditional_prefill_groups':sum(not c['all_M3'] for c in certs),'conditional_small_row_groups':sum(c['all_M3'] for c in certs),'conditional_proxy_bounds':{'sum_current_max_rank_rows':tot_current_rows,'sum_whole_expert_unsplit_row_LB':tot_lower_rows,'sum_average_row_only_LB':tot_avg_rows,'row_proxy_improvement_cap':tot_current_rows/tot_lower_rows,'naive_average_row_proxy_cap':tot_current_rows/tot_avg_rows,'sum_current_max_active_experts':tot_current_active,'sum_active_expert_LB':tot_lower_active,'active_reference_proxy_improvement_cap':tot_current_active/tot_lower_active,'row_group_caps':stats([g['whole_expert_row_proxy_improvement_cap'] for g in certs if g['whole_expert_row_proxy_improvement_cap'] is not None])},
 'conditional_static_replication_LB':{'layers':len(static),'original_pairs_per_layer':384,'minimum_full_pair_copies_total':sum(x['minimum_static_full_weight_pair_copies'] for x in static),'original_pairs_total':384*len(static),'extra_pairs_total':sum(x['extra_pairs_above_original384'] for x in static),'full_pair_int8_bytes':W,'extra_int8_weight_bytes_LB':sum(x['extra_int8_weight_bytes_LB'] for x in static),'extra_int8_weight_GiB_LB':sum(x['extra_int8_weight_bytes_LB'] for x in static)/1024**3},
 'guard_refutation_domain':'Mask is BOOL;fixed expert ownership/current MC2 semantic/protocol contract. Zero local senders does not imply zero incoming expert work;zero receivers does not imply zero outgoing/return work. Coordinated protocol/placement replacement is a different domain.',
 'proofs':{'whole_expert':'For sorted indivisible loads w_k on16ranks,every placement has maxload>=ceil(sumw/16) and >=ceil(k/16)*w_k for eachk. Current/lower bounds possible row-proxy improvement;no feasible schedule inverted.','active_expert':'Each active expert remains onone owner;maxactive>=ceil(A/16). This is referenced-weight count,notHBM bytes or time.','replication':'To constrain every rank to atmostbrows,expert e with n_e rows needs >=ceil(n_e/b) full replicas in the row-splitting/full-pair domain. A fixed layout across calls needs c_e>=max(1,max_tceil(n_et/b_t)). Count all original experts;no compression,eviction,cross-expert value dedup or axis sharding in thisdomain.'},
 'limits':['All crossrank bounds remain conditional on the separately recorded semantic-group/order identity premises.','Proxy improvement caps are not latency/throughput/physical-speedup caps.','MACwork is directdense/source-domain necessary work;physical P_MACupper andproduction MFU not supplied.','Cumulative rank/kernel time is not request latency or cluster throughput.','Zero localGEMMwork does not identify root cause:other ranks compute/memory,vector work,network andprotocol waits remain competitors.','Replica byte lower bound does not establish memory infeasibility without matched usable memory/other-residency evidence.','Diagnostic observers add readers and overhead;no production speedup claim.'],'historical_native_or_phase_promotions':0,'new_whole_module_closures':0,'overall_complete':False}
(O/'ALL-TARGET-PLACEMENT-BOUNDS.json').write_text(json.dumps(summary,indent=2)+'\n');(O/'ALL-TARGET-PLACEMENT-CERTIFICATES.jsonl').write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in certs));(O/'ALL-TARGET-STATIC-REPLICA-LOWER-BOUNDS.json').write_text(json.dumps(static,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['actual_guard_examples','proofs','limits']}))
