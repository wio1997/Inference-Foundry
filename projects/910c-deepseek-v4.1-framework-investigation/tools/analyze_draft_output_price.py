#!/usr/bin/env python3
"""Price whole base-head/Markov/ID delivery and paid nested alternatives."""
from pathlib import Path
from hashlib import sha256
from statistics import mean,median
import json,zlib
from analyze_o_projection_parent_price import start,dur,union,first_shape
R=Path('audit/execution-search');B=Path('/Users/wio/work/branch-fusion-audit-20261104/full-review')
def main():
 paths=[R/'draft-output-parent-primary.json',R/'target-vocab-parent-primary.json',R/'draft-output-dependencies.json',R/'swa-publication-parent-calls.json.zlib',R/'qkv-joint-delivery-price.json',B/'MARKOV-BIAS-CACHE.json',B/'MARKOV-OWNER-GROUPS.json',B/'DSPARK-CURRENT-OUTPUT-PUBLICATION.json',B/'ORDERED-SAMPLING-VALUE.json']
 raw=json.loads(zlib.decompress(paths[3].read_bytes()));windows=json.loads(paths[4].read_text());profiles=[]
 for src in raw:
  by={r['csv_row']:r['data'] for r in src['selected']};ids=sorted(by);mm=[i for i in ids if by[i]['Type']=='MatMulV2' and by[i]['Input Shapes'].strip('"').split(';')==['4,256','129280,256']];assert len(mm)==60
  chains=[]
  for k in range(0,len(mm),5):
   steps=mm[k:k+5];stream=by[steps[0]]['Stream ID'];assert all(by[i]['Stream ID']==stream for i in steps)
   heads=[i for i in ids if steps[0]-20<i<steps[0] and by[i]['Stream ID']==stream and by[i]['Type']=='MatMulV2' and by[i]['Input Shapes'].strip('"').split(';')==['20,5120','16160,5120']];head,=heads
   norm=head-1;assert by[norm]['Type']=='RmsNorm';gather=head+1;assert by[gather]['Type']=='hcom_allGather_';transpose=head+2;assert by[transpose]['Type']=='Transpose'
   tasks=[];post=[];layout=[]
   for n,i in enumerate(steps):
    part=[j for j in ids if i-2<=j<=i+8 and by[j]['Stream ID']==stream];types=[by[j]['Type'] for j in part];assert types==['Slice','GatherV2','MatMulV2','Slice','Add','ViewCopy','Slice','Slice','ArgMaxV2','Cast','ViewCopy'],types
    assert 'DT_BF16' in by[i]['Output Data Types'] and 'DT_BF16' in by[i+2]['Output Data Types'];tasks+=part;post+=part[3:];layout+=part[3:6]
   assert len(tasks)==55 and len(set(tasks))==55
   end=start(by[steps[-1]+8])+dur(by[steps[-1]+8]);begin=start(by[norm]);loopbegin=start(by[tasks[0]])
   assert union([by[i] for i in tasks])<=end-loopbegin
   w2=sum(float(dur(by[i])) for i in steps);postsum=float(union([by[i] for i in post]));g=float(dur(by[gather]));loopnative=float(union([by[i] for i in tasks]));hostgap=float(end-loopbegin)-loopnative
   scenarios=[]
   for variant,eta,comm,combined in [('replicated_post_fusion',0,0,20),('vocab_sharded',.5,10,20),('vocab_sharded',.875,10,20),('vocab_sharded',.875,30,20),('vocab_sharded',.875,100,20)]:
    net=eta*w2+postsum+(g if variant=='vocab_sharded' else 0)-5*(comm+combined)
    scenarios.append(dict(variant=variant,fraction_W2_time_removed=eta,each_new_small_collective_us=comm,each_new_post_kernel_us=combined,local_net_us=net,qualification='Sensitivity only: retained original base GEMM, W2 remaining time paid; original post math/ID writes live in new post kernel. Replicated variant retains full base gather. eta.875 is ideal8-way logical work scenario, not achieved timing; sharding pays five serial rounds. Host gaps and cache hits uncredited.'))
   previous_post=max(i for i in ids if head-70<i<head and by[i]['Stream ID']==stream and by[i]['Type']=='HcPost');ready=start(by[previous_post])+dur(by[previous_post]);assert previous_post<norm and end>ready
   chains.append(dict(head_csv_row=head,markov_csv_rows=steps,final_ID_write=steps[-1]+8,previous_draft_Post=previous_post,main_stream=stream,base_head_us=float(dur(by[head])),base_gather_us=g,base_transpose_us=float(dur(by[transpose])),whole_Post_ready_to_ID_done_us=float(end-ready),whole_norm_to_ID_done_us=float(end-begin),whole_Markov_to_ID_done_us=float(end-loopbegin),Markov_native_us=loopnative,Markov_uncovered_span_us=hostgap,W2_us=w2,post_W2_all_tasks_us=postsum,corrected_writeback_tasks_us=float(union([by[i] for i in layout])),scenarios=scenarios))
  w=next(x for x in windows['profiles'] if x['ep']==src['ep']);sums={k:sum(x[k] for x in chains) for k in chains[0] if k.endswith('_us')};profiles.append(dict(ep=src['ep'],window_us=w['window_us'],calls=len(chains),sum_us=sums,window_pct={k:v/w['window_us']*100 for k,v in sums.items()},median_us={k:median(x[k] for x in chains) for k in sums},scenario_totals=[dict(**{k:v for k,v in chains[0]['scenarios'][j].items() if k!='local_net_us'},net_us=sum(x['scenarios'][j]['local_net_us'] for x in chains),net_window_pct=sum(x['scenarios'][j]['local_net_us'] for x in chains)/w['window_us']*100) for j in range(5)],chains=chains))
 cache=json.loads(paths[5].read_text());current={s['runtime_path']:s['runtime_sha256'] for s in json.loads(paths[2].read_text())['sources']};matches=[dict(path=k,sha256=v,current_equal=current.get(k)==v) for k,v in cache['native']['source_sha256'].items()];assert all(x['current_equal'] for x in matches)
 native=dict(source_matches=matches,checkpoint=cache['native']['checkpoint'],limited_checks=cache['native']['checks'],allhit_saved_us=cache['timing_and_break_even']['all_hit_saved_us'],mixed_penalty_us=cache['timing_and_break_even']['mixed_penalty_us_by_miss_count_1_2_3'],old_required_all_hit_step_fraction=cache['timing_and_break_even']['all_hit_fraction_to_offset_each_mixed_penalty'],usable_saved_predecessor_IDs=cache['saved_id_availability']['usable_predecessor_IDs'],qualification=cache['native']['scope']+' Four current Python source SHAs match; no fresh native execution, loaded-binary attestation or universal N-shard/shape proof. Do not repeat unchanged allhit/mixed fixtures; workload allhit rate and paid classifier/cache mutation remain missing.')
 publication=json.loads(paths[7].read_text());scope_req={}
 for name,data in publication['current_carrier_sensitivity'].items():
  if 'Markov' in name or 'draft' in name or 'target' in name:
   scope_req[name]={kind:dict(upper_ms=v['upper_ms'],max_request_percent=max(r['max_current_credit_ms']/r['TTFT_ms']*100 for r in v['requests'])) for kind,v in data.items()}
 out=dict(status='MATCHED_WHOLE_DRAFT_OUTPUT_AND_PAID_ALTERNATIVES_WITH_HISTORICAL_NATIVE_INTERSECTION',profiles=profiles,mean_window_pct={k:mean(p['window_pct'][k] for p in profiles) for k in profiles[0]['window_pct']},native_cache_history=native,current_carrier_request_qualification=scope_req,request_scope=publication['scope'],request_limits=publication['limits'],request_credit=0,
  limits=['Only old EP0/EP8 mirrored C8 B4,K5,R256,V129280 windows for local sums; not full16rank/request or sixordinary timing. Native W2 scalar/source identities are a separate historical fixture.','Whole spans include necessary base/W2 math, copying and exposed gaps; gaps are not proved host-only or free. New math/collectives/cache misses/classifier/format costs must be paid.','Fixed target vocab gather/layout summary and terminal tiny recipes are historical stopped children, not a cap on full base-head/Markov/verification parent.','Current-carrier request envelopes freeze every earlier iteration/start and end at the next measured Fill, not CPU/D2H/client completion; do not sum them with local fractions or claim actual request speedup.','Cache allhit/mixed native tests have no workload ID population: saved placeholder IDs cannot estimate cache hit rate; unknown request/cache gain defaults0.'],inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in paths],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
 (R/'draft-output-price.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(local_mean_pct=out['mean_window_pct'],scenarios=[p['scenario_totals'] for p in profiles],source_matches=4,usable_IDs=native['usable_saved_predecessor_IDs'],request_envelope_scopes=list(scope_req)),ensure_ascii=False))
if __name__=='__main__':main()
