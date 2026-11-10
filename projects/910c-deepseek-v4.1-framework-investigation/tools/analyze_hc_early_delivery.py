#!/usr/bin/env python3
"""Price the premix-present HC fork/join with retained coefficient work, no NPU."""
from pathlib import Path
from hashlib import sha256
import json,zlib,itertools,statistics
from analyze_o_projection_parent_price import start,dur

ROOT=Path('audit/execution-search')
def main():
 rawpath=ROOT/'swa-publication-parent-calls.json.zlib';parentpath=ROOT/'o-projection-parent-price.json'
 raw=json.loads(zlib.decompress(rawpath.read_bytes()));parents=json.loads(parentpath.read_text());profiles=[]
 for p in parents['profiles']:
  src=next(r for r in raw if r['ep']==p['ep']);byid={r['csv_row']:r['data'] for r in src['selected']};ids=sorted(byid);records=[];used=set()
  for r in p['deliveries']:
   tail=[i for i in ids if r['RmsNormCast']<i<r['RmsNormCast']+32 and byid[i]['Stream ID']==r['main_stream']]
   post=next(i for i in tail if byid[i]['Type']=='HcPost')
   dfc=[i for i in tail if i<post and byid[i]['Type']=='DispatchFFNCombine'];assert len(dfc)==1
   assert post not in used;used.add(post)
   assert byid[post]['Input Shapes'].strip('"').split(';')[0]=='1,3,5120'
   a=byid[r['HcPre']];n=byid[r['RmsNormCast']];end=start(byid[post]);prestart=start(a)
   c=float(dur(a));norm=float(start(n)+dur(n)-start(a)-dur(a));mlp=float(end-start(n)-dur(n))
   assert norm>=0 and mlp>=0
   records.append(dict(rows=r['rows'],HcPre=r['HcPre'],RmsNormCast=r['RmsNormCast'],DFC=dfc[0],FFNHcPost=post,HcPre_us=c,norm_and_gap_us=norm,MLP_to_FFNPost_us=mlp,original_pre_to_FFNPost_us=float(end-prestart)))
  scenarios=[]
  for rows,e,delta,h,gamma in itertools.product([24,20],[5,10,20],[0,10,40],[0,5],[0,.5]):
   vals=[r for r in records if r['rows']==rows];new=[]
   for r in vals:
    # Keep the complete original HCPre for coefficients, including its old collapse;
    # a separate early original-order collapse+norm duplicates a small amount of work.
    coeff=r['HcPre_us']*(1+gamma)
    main=e+r['norm_and_gap_us']+r['MLP_to_FFNPost_us']+delta
    new.append(max(coeff,main)+h)
   old=sum(r['original_pre_to_FFNPost_us'] for r in vals);net=old-sum(new)
   scenarios.append(dict(rows=rows,early_collapse_extra_us=e,MLP_resource_slowdown_us=delta,additional_nonoverlap_us=h,coefficient_branch_slowdown_fraction=gamma,retained_original_HCPre=True,conditional_net_us=net,conditional_window_pct=net/p['window_us']*100))
  groups=[]
  for rows in [24,20]:
   vals=[r for r in records if r['rows']==rows]
   groups.append(dict(rows=rows,deliveries=len(vals),HCPre_us_sum=sum(r['HcPre_us'] for r in vals),MLP_us_minmax=[min(r['MLP_to_FFNPost_us'] for r in vals),max(r['MLP_to_FFNPost_us'] for r in vals)],original_parent_us_sum=sum(r['original_pre_to_FFNPost_us'] for r in vals),HCPre_exposure_ceiling_us=sum(r['HcPre_us'] for r in vals),HCPre_exposure_ceiling_window_pct=sum(r['HcPre_us'] for r in vals)/p['window_us']*100))
  profiles.append(dict(ep=p['ep'],window_us=p['window_us'],source_consistent_chains=len(records),groups=groups,scenarios=scenarios,deliveries=records))
 result=dict(status='CONDITIONAL_PREMIX_PRESENT_EARLY_COLLAPSE_FORK_JOIN_SCREEN',profiles=profiles,
  original_dependencies=['rounded_attn_HCPost -> whole FFNHCPreV3 -> norm -> gate/route/DFC/shared -> FFNHCPost','current FFNHCPreV3 emits newpre/post/comb but ProcessY uses existingattn_pre whenhasPreMix=1','FFNHCPost needs rounded residual, currentFFN post/comb and FFNresult; nextdecoder gets currentFFNpre'],
  proposed_dependencies=['rounded_HCPost -> original_order_collapse(existingattn_pre) -> original_order_norm -> MLP','rounded_HCPost -> retained_original_HCPreV3 coefficient branch -> currentpost/comb/pre','join at FFNHCPost and before nextdecoder pre_mix consumer; no wholeHCPre wait on earlynorm branch'],
  assumptions=['Only hasPreMix=1. No-premix first attention collapse depends on newlycomputedpre; original fallback mandatory.','Early branch extra e includes duplicatecollapse/ownlaunch resource demand beyond retained originalnorm. Added nonoverlap h and MLP slowdown delta paid separately; coefficient slowdown gamma retained.','Retain entire original HCPre coefficient branch including oldunusedY as conservative non-elimination schedule. Does not claim original24AIC/48AIV overlaps freely with gate/DFC.','All scenarios are analytical fork/join sensitivities from two matched saved windows; no activeprofile dependency/pointer proof or requestprediction.','Actual native bit, FULLgraph splitoutput event/capture, crossrank collective launch order, completebuffer/flag and fullrequest tests still pending.','Newpost/collapse/norm fusion must retain BF16HCPost, originalFP32 fourterm leftfold and BF16collapse before originalnorm; FP32routing widens finalBF16, not pre-roundFP32.'],
  new_primitive_admissions=0,whole_chain_complete=False,every_replacement_bitwise=True,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],inputs=[dict(path=str(f),sha256=sha256(f.read_bytes()).hexdigest()) for f in [rawpath,parentpath]],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
 (ROOT/'hc-early-delivery-price.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps([dict(ep=p['ep'],groups=p['groups'],target_e5_delta0_h5_gamma0=[s for s in p['scenarios'] if s['rows']==24 and s['early_collapse_extra_us']==5 and s['MLP_resource_slowdown_us']==0 and s['additional_nonoverlap_us']==5 and s['coefficient_branch_slowdown_fraction']==0],target_e5_delta40_h5_gamma0=[s for s in p['scenarios'] if s['rows']==24 and s['early_collapse_extra_us']==5 and s['MLP_resource_slowdown_us']==40 and s['additional_nonoverlap_us']==5 and s['coefficient_branch_slowdown_fraction']==0]) for p in profiles],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
