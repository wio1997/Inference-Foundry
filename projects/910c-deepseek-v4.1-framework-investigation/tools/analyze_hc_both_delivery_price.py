#!/usr/bin/env python3
"""Bind both HCPre parents and price independent output schedules with paid joins."""
from pathlib import Path
from hashlib import sha256
import json,zlib,itertools
from analyze_o_projection_parent_price import start,dur,first_shape

ROOT=Path('audit/execution-search')
def main():
 rawpath=ROOT/'swa-publication-parent-calls.json.zlib';qfile=ROOT/'qkv-joint-delivery-price.json';ofile=ROOT/'o-projection-parent-price.json';ffile=ROOT/'hc-early-delivery-price.json'
 raw=json.loads(zlib.decompress(rawpath.read_bytes()));qs=json.loads(qfile.read_text());ops=json.loads(ofile.read_text());ffns=json.loads(ffile.read_text());profiles=[]
 for p in qs['profiles']:
  src=next(r for r in raw if r['ep']==p['ep']);byid={r['csv_row']:r['data'] for r in src['selected']};ids=sorted(byid)
  op={r['smla']:r for r in next(x for x in ops['profiles'] if x['ep']==p['ep'])['deliveries']};ff={r['HcPre']:r for r in next(x for x in ffns['profiles'] if x['ep']==p['ep'])['deliveries']};recs=[];used=set()
  for r in p['deliveries']:
   before=[i for i in ids if r['preceding_input_ag']-32<i<r['preceding_input_ag'] and byid[i]['Stream ID']==r['main_stream']]
   norm=max(i for i in before if byid[i]['Type']=='RmsNorm');pre=max(i for i in before if i<norm and byid[i]['Type']=='HcPre');o=op[r['smla']];f=ff[o['HcPre']]
   assert pre not in used;used.add(pre)
   assert first_shape(byid[pre])=='3,4,5120' and first_shape(byid[norm])=='3,5120'
   assert byid[pre]['Input Shapes'].strip('"').split(';')[-1]=='3,4'
   c=float(dur(byid[pre]));n=float(start(byid[norm])+dur(byid[norm])-start(byid[pre])-dur(byid[pre]));a=float(start(byid[o['HcPost']])-start(byid[norm])-dur(byid[norm]));assert min(c,n,a)>=0
   pmu={k:{engine:(float(byid[i][engine]) if byid[i][engine].strip() not in ['', 'N/A'] else None) for engine in ['aicore_time(us)','aiv_time(us)']} for k,i in [('attention',pre),('ffn',f['HcPre'])]}
   recs.append(dict(rows=r['rows'],attention_HCPre=pre,attention_RMSNorm=norm,attention_Post=o['HcPost'],ffn_HCPre=f['HcPre'],ffn_Post=f['FFNHcPost'],attention_HCPre_us=c,attention_norm_and_gap_us=n,attention_norm_end_to_Post_us=a,ffn_HCPre_us=f['HcPre_us'],ffn_norm_and_gap_us=f['norm_and_gap_us'],ffn_downstream_to_Post_us=f['MLP_to_FFNPost_us'],saved_HCPre_PMU_us=pmu))
  groups=[];scenarios=[];pressure=[]
  for rows in [24,20]:
   vals=[r for r in recs if r['rows']==rows];ac=sum(r['attention_HCPre_us'] for r in vals);fc=sum(r['ffn_HCPre_us'] for r in vals)
   pmu_sums={k:{engine:dict(available=sum(r['saved_HCPre_PMU_us'][k][engine] is not None for r in vals),sum_us=sum(r['saved_HCPre_PMU_us'][k][engine] for r in vals if r['saved_HCPre_PMU_us'][k][engine] is not None)) for engine in ['aicore_time(us)','aiv_time(us)']} for k in ['attention','ffn']}
   groups.append(dict(rows=rows,deliveries=len(vals),attention_HCPre_us=ac,ffn_HCPre_us=fc,both_HCPre_us=ac+fc,both_coefficient_hide_ceiling_window_pct=(ac+fc)/p['window_us']*100,attention_downstream_minmax_us=[min(r['attention_norm_end_to_Post_us'] for r in vals),max(r['attention_norm_end_to_Post_us'] for r in vals)],saved_PMU_sums_us=pmu_sums))
   for e,delta,h,gamma in itertools.product([5,10,20],[0,10,40],[0,5],[0,.5]):
    nets={k:0. for k in ['attention','ffn']}
    for r in vals:
     for key,c,n,a in [('attention',r['attention_HCPre_us'],r['attention_norm_and_gap_us'],r['attention_norm_end_to_Post_us']),('ffn',r['ffn_HCPre_us'],r['ffn_norm_and_gap_us'],r['ffn_downstream_to_Post_us'])]:
      nets[key]+=c+n+a-(max(c*(1+gamma),e+n+a+delta)+h)
    scenarios.append(dict(rows=rows,early_collapse_extra_us_each=e,downstream_resource_slowdown_us_each=delta,additional_nonoverlap_us_each=h,coefficient_branch_slowdown_fraction=gamma,retained_whole_original_coefficient_kernels=True,attention_net_us=nets['attention'],ffn_net_us=nets['ffn'],combined_net_us=sum(nets.values()),conditional_combined_window_pct=sum(nets.values())/p['window_us']*100))
   for e,metric in itertools.product([5,10],['aicore_time(us)','max_recorded_engine_time']):
    net=0.;proxy=0.;covered=0;missing=[]
    for r in vals:
     for key,c,n,a in [('attention',r['attention_HCPre_us'],r['attention_norm_and_gap_us'],r['attention_norm_end_to_Post_us']),('ffn',r['ffn_HCPre_us'],r['ffn_norm_and_gap_us'],r['ffn_downstream_to_Post_us'])]:
      pmu=r['saved_HCPre_PMU_us'][key]
      if (metric=='max_recorded_engine_time' and any(v is None for v in pmu.values())) or (metric!='max_recorded_engine_time' and pmu[metric] is None):missing.append(dict(kind=key,csv_row=r[key+'_HCPre']));continue
      delta=max(pmu.values()) if metric=='max_recorded_engine_time' else pmu[metric];proxy+=delta;covered+=1
      net+=c+n+a-(max(c,e+n+a+delta)+5)
    pressure.append(dict(rows=rows,early_collapse_extra_us_each=e,additional_nonoverlap_us_each=5,contention_proxy=metric,covered_HCPre_calls=covered,total_HCPre_calls=2*len(vals),missing_PMU_calls=missing,total_slowdown_proxy_us=proxy,covered_calls_conditional_net_us=net,covered_calls_window_pct=net/p['window_us']*100,uncovered_net_us=None,qualification='Covered-call sensitivity only, no fullcandidate prediction or runtime free-fallback selector inferred from profiler rows. Missing PMU remains unknown, never zero. Saved engine field is not actual newslowdown, exclusivephase, occupancy or a bound. Do not add AIC/AIV fields.'))
  profiles.append(dict(ep=p['ep'],window_us=p['window_us'],source_consistent_pairs=len(recs),groups=groups,scenarios=scenarios,pressure_proxy_scenarios=pressure,deliveries=recs))
 result=dict(status='BOTH_PREMIX_HC_PARENTS_RETAIN_ORIGINAL_WORK_CONDITIONAL_FORK_JOIN',profiles=profiles,
  actual_source_eligibility='Target and DSpark seed FP32pre_mix [1,0,0,0] before firstdecoder; every subsequent decoder receives previousFFNpre; each FFN receives sameblockattnpre. Both call V3 with pre_mix tensor. Public ABI None remains fallback, not an actual firstlayer in these two forwards.',
  source_correction=dict(previous='Earlier hc-early-delivery-price assumed no-premix firstattention as fallback example.',qualification='Generic None fallback is real, but must not describe actual target/DSpark firstattention: both explicit model entry seeds are non-None. Historical artifact unchanged, current source correction supersedes firstlayer interpretation.'),
  limits=['Exact2parent/rank source-consistent binding, not pointer/event proof. No proof of fullrequest, ordinary or inactiveDP/capture cases.','Both branch prices explicitly recomputed from two serial HC parents with separate e/h/delta/gamma perbranch; no free simultaneousresource assumption.','Recorded aicore_time/aiv_time are not exclusive phase durations or available concurrent capacity. Pressureproxy cases deliberately use them as sensitivities, never as actual newkernel contention or a bound; do not add AIC and AIV times.','Joined before each own HCPost, so original coefficient pre/post/comb remains complete before nextdecoder, aux snapshot, Engram update and finalcollapse under this conservative protocol. Finer output events need separate proof.','Do not add these percentages to earlier QKV/head/Oprojection/frame plans; those change downstream/overlap/resource and require a combined paid schedule.','Whole currentHCPre is retained as conservative baseline, including duplicateunusedY and208MiB workspace. No parameter or live output deletion/VRAM credit.','Cost scenarios, not implementation/bit/requestprediction. Original input RMS remains originalop, FFN norm/cast remains roundedBF16->FP32.'],
  whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],inputs=[dict(path=str(f),sha256=sha256(f.read_bytes()).hexdigest()) for f in [rawpath,qfile,ofile,ffile]],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
 (ROOT/'hc-both-delivery-price.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps([dict(ep=p['ep'],groups=p['groups'],target_pressure=[s for s in p['pressure_proxy_scenarios'] if s['rows']==24],target_e5_delta0_h5_gamma0=[s for s in p['scenarios'] if s['rows']==24 and s['early_collapse_extra_us_each']==5 and s['downstream_resource_slowdown_us_each']==0 and s['additional_nonoverlap_us_each']==5 and s['coefficient_branch_slowdown_fraction']==0]) for p in profiles],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
