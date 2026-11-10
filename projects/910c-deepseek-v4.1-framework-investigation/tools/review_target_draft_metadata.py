#!/usr/bin/env python3
"""Bound source side effects and price one complete target-only delivery.

Original Python control bodies run with opaque callbacks, never Torch/NPU.
"""
import ast, copy, itertools, json, statistics
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace as NS
ROOT=Path('audit/execution-search')
def main():
 deps=json.loads((ROOT/'target-draft-metadata-dependencies.json').read_text());graph=json.loads((ROOT/'target-draft-graph-dependencies.json').read_text());sources=deps['sources']+graph['sources'];excerpts=[]
 for s in sources:assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
 def method(suffix,owner,name,env):
  s=next(s for s in sources if s['runtime_path'].endswith(suffix));p=Path(s['local_path']);ls=p.read_text().splitlines();cls=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name==owner);n=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name));n.decorator_list=[]
  excerpts.append(dict(runtime_path=s['runtime_path'],sha256=s['runtime_sha256'],owner=owner,method=name,first=n.lineno,last=n.end_lineno,lines=ls[n.lineno-1:n.end_lineno]))
  mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),str(p),'exec'),env);return env[name]
 controls=[]
 resolve=method('/attention/context_parallel/dsa_v41_cp.py','_ReplicatedCacheImpl','_global_layer_metadata',{})
 class PrefixMap(dict):
  def __iter__(self):raise AssertionError('whole map iteration forbidden in prefix resolver')
  def values(self):raise AssertionError('whole map values forbidden in prefix resolver')
  def items(self):raise AssertionError('whole map items forbidden in prefix resolver')
 for bits in itertools.product([False,True],repeat=3):
  prefixes=['target.swa']+[f'target.plane{i}' if enabled else None for i,enabled in enumerate(bits)]
  impl=NS(swa_prefix=prefixes[0],long_kv_source_prefix=prefixes[1],index_k_source_prefix=prefixes[2],compressor_state_prefix=prefixes[3],_get_layer_metadata=lambda x:x)
  target={p:NS(global_metadata=object()) for p in prefixes if p is not None};baseline=PrefixMap(target);baseline.update({f'mtp.{i}.self_attn.swa_cache':object() for i in range(3)})
  a=resolve(impl,baseline);b=resolve(impl,PrefixMap(target));assert a==b
  controls.append(dict(case='target_prefix_resolver',optional_planes=list(bits),same_exact_global_objects=True,draft_entries_read=False))
 g={};draft=method('/attention/dsa_v41.py','AscendDSAV41MetadataBuilder','build_for_drafting',g)
 for phase in ['after_target','after_skipped_target']:
  seen=[];common=object();builder=NS(build=lambda *a,**k:seen.append((a,k)) or 'fresh')
  assert draft(builder,common,1,common_v41_batch_metadata={'stale':'must_not_forward'})=='fresh'
  assert seen==[((0,common),{})];controls.append(dict(case='fresh_draft_build_kwargs',phase=phase,args_preserved=True,shared_target_kwargs_forwarded=False))
 g={'DeviceMetadataStage':NS(COMPRESSOR='C'),'DeviceMetadataTask':lambda *a:NS(stage=a[0],run=a[1],group_id=a[2])}
 publish=method('/attention/dsa_v41.py','AscendDSAV41MetadataBuilder','_publish_task',g);take=method('/attention/dsa_v41.py','AscendDSAV41MetadataBuilder','take_device_metadata_tasks',g)
 for tokens,full in itertools.product([0,24],[False,True]):
  oldpacket=object();own=object();seen=[];b=NS(_device_metadata_enabled=True,_device_metadata_in_graph=False,_device_metadata_tasks=())
  assert publish(b,{'smla:c0':oldpacket},'smla:c0',own,'A',lambda:seen.append('native')) is oldpacket
  assert take(b)==() and not seen
  controls.append(dict(case='last_swa_packet_cache_hit',actual_tokens=tokens,full=full,earlier_packet_identity_preserved=True,new_tasks=0,qualification='Token/full labels do not execute native arithmetic; original publish cache-hit body is input independent.'))
 # A miss is a real producer; never skip if the preceding first owner is absent.
 b=NS(_device_metadata_enabled=True,_device_metadata_in_graph=False,_device_metadata_tasks=());seen=[]
 publish(b,{},'smla:c0',object(),'A',lambda:seen.append('producer'));tasks=take(b);assert len(tasks)==1;tasks[0].run();assert seen==['producer'] and take(b)==()
 controls.append(dict(case='missing_first_owner_negative',new_tasks=1,elision_requires_preceding_owner=True))
 b._device_metadata_tasks=('stale-from-failed-build',);assert take(b)==('stale-from-failed-build',)
 controls.append(dict(case='skip_build_keep_take_negative',stale_task_would_be_returned=True,consequence='Reset before a retained take, or remove the unused build/take pair together and prove retained tuples are never submitted/read before the next reset. Normal drained history alone does not establish failure recovery.'))
 g={'CUDAGraphMode':NS(FULL='FULL'),'torch':NS(npu=NS(current_stream=lambda:(_ for _ in ()).throw(AssertionError('no stream expected')))),'update_full_graph_params':lambda *a:(_ for _ in ()).throw(AssertionError('not selected'))}
 update=method('/worker/model_runner_v1.py','NPUModelRunner','_update_full_graph_params_if_needed',g)
 for mode,capturing,sparse,enpu in itertools.product(['FULL','NONE'],[False,True],[False,True],[False,True]):
  update(NS(use_sparse=sparse,use_compress=True,enable_enpu=enpu),NS(cudagraph_runtime_mode=mode,capturing=capturing),24)
  controls.append(dict(case='legacy_full_graph_update_guard',mode=mode,capturing=capturing,sparse=sparse,enpu=enpu,use_compress=True,legacy_metadata_update_called=False))
 tail=method('/worker/model_runner_v1.py','NPUModelRunner','_maybe_eager_restore_copy_sfa_tails',{})
 for need in [False,True]:
  b=NS(_copy_sfa_need_eager_tail_restore=need);tail(b,{'target':NS(),'draft':NS()});assert not b._copy_sfa_need_eager_tail_restore
  controls.append(dict(case='no_copy_sfa_metadata_attributes',need=need,restore_called=False,qualification='Actual offload enabled state separate; original predicate checks fields absent from V41 metadata.'))
 raw=json.loads((ROOT/'target-draft-metadata-price-raw.json').read_text());rs=raw['ranks'];assert len(rs)==16 and all(len(r['calls'])==11 for r in rs)
 means={k:statistics.mean(r['totals_ms'][k] for r in rs) for k in rs[0]['totals_ms']};window=statistics.mean(r['window_ms'] for r in rs)
 reference=json.loads((ROOT/'common-device-price.json').read_text())['reference_light'];assert reference['metadata_frame_counts_by_DP']=={'0':[229],'1':[231]};sec=reference['batch_seconds'];scenarios=[]
 for p,rho,h in [(0,0,.1),(.5,.5,.05),(.5,.5,.1),(1,.5,.1),(1,1,0)]:
  vals=[]
  for r in rs:
   n=reference['metadata_frame_counts_by_DP'][str(r['ep']//8)][0];vals.append(n*(p*rho*r['totals_ms']['guarded_application_ms']/11-h))
  scenarios.append(dict(eligible_frame_fraction=p,diagnostic_to_request_transfer=rho,new_ms_per_reference_frame=h,conditional_mean_ms=statistics.mean(vals),conditional_min_ms=min(vals),conditional_max_ms=max(vals),conditional_mean_batch_pct=statistics.mean(vals)/(sec*1000)*100))
 price=dict(status='WORTH_DEEPENING_WHOLE_TARGET_DRAFT_METADATA',cohort='Saved detailed balanced C8 32K/1K; 16 ranks x 11 complete target frames',calls=176,window_mean_ms=window,mean_11_frame_totals_ms=means,diagnostic_window_pct={k:v/window*100 for k,v in means.items()},body_call_median_ms=statistics.median(c['body_ns']/1e6 for r in rs for c in r['calls']),guarded_call_median_ms=statistics.median(c['guarded_application_ms'] for r in rs for c in r['calls']),original_unmatched=sum(len(r['unmatched']) for r in rs),reference_light=reference,request_scenarios=scenarios,request_formula='Per rank: N_DP*(eligible_fraction*diagnostic_transfer*guarded_Free_ms_per_frame - new_ms_per_reference_frame). Rank range/mean are conditional proxies, not a simulation of TP critical path or an observed speedup.',default_unconfirmed_eligible_fraction=0,default_unconfirmed_transfer=0,qualification=raw['qualification'],not_additive_with=['Prior common CPU helper reuse','Prior last25 local-length dead projection','Larger input metadata whole frame'],not_qualified=['original first C8 prompt/cohort','six ordinary loads','prefill/TTFT','queued work after parent return','unprofiled net gain'],inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [ROOT/'target-draft-metadata-primary.json',ROOT/'target-draft-metadata-price-raw.json',ROOT/'common-device-price.json']])
 price['local_net_scenarios']=[dict(real_deletion_and_exposure_fraction=rho,new_ms_per_frame=h,conditional_11_frame_ms=means['guarded_application_ms']*rho-11*h,conditional_diagnostic_window_pct=(means['guarded_application_ms']*rho-11*h)/window*100) for rho,h in [(0,.1),(.5,.05),(.5,.1),(1,.1)]]
 (ROOT/'target-draft-metadata-price.json').write_text(json.dumps(price,ensure_ascii=False,indent=2)+'\n')
 review=dict(status='待确认',introduction='Target-phase gid12 entire CP/global/local metadata delivery; preserve CM/rawphysical setters and actual draft producer.',reached=['15 current source pins:13+2, not15 full files reviewed.','Original target prefix resolver reads only its own four plane prefixes; three draft dictionary entries have no direct reader there.','Target gid12 is last and cache-kind SWA; global/local batch and RoPE shared keys already first owned earlier. No gid12 C2/ring producer; local smla:c0 publication is a hit if earlier target SWA owner exists.','Legacy FULL params update is excluded by use_compress. Generic ACL graph may still replay captured tensor addresses: this guard alone does not prove graph unreadness.','Actual drafting reconstructs with fresh kwargs/noncausal metadata; group raw physical views are populated outside the candidate.','Removing the target build also removes task-tuple resets. A synthetic stale-tuple negative control rejects skipping build while retaining take without its matching reset; removing both requires a separate read/submission proof.'],result='Whole body has material diagnostic Free exposure, so retain for implementation investigation; no primitive/framework admission. Do not prune this parent using previous native-only island price.',controls=controls,remaining=['Trace selected ACLGraph runtime backend predicate/graph-task registrations and all captured draft-buffer reads; legacy update bypass is insufficient.','Check global/local persistent QSL/slots/seq buffers from init through target/skipped-target/draft/capture/error, all live read spans and overwrite-before-read. Dead private tails alone need not be preserved, but deadness must be proved.','Retain CM selection, unpadding, per-group table/slot setter; define target metadata dictionary key/schema change and generic MM/offload/logging readers without aliasing another physical group.','Reset both task tuples before a retained take, or remove the unused build/take pair and prove no tuple read/submission before the next reset; prove first-owner availability and error recovery including empty/synthetic DP.','Native complete packet/retained-output bit validation and exact shape/address/capture qualification before framework installation; each replacement primitive bitwise.','Unprofiled net cost and matched eligible population/critical-path transfer; original C8/prefill/six ordinary whole-parent prices remain unqualified.'],stopped=['Skip the whole last KV group before CM/rawphysical setter (would delete live draft input delivery).','Skip build but retain take without its matching reset/protocol proof.','Treat target packet cache-hit as permission to delete the earlier packet producer.','Claim original native-small-island price is a cap on this complete host builder.'],confirmed_implementable_new=0,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,original_static_source_bindings_unchanged=90,original_static_sites_pending_unchanged=66,qualification='Opaque original-Python controls/source facts and old diagnostic intervals only. Not Torch/native equality, live graph read-set, runtime race evidence or measured net speedup.',sources=sources,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],reviewer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
 (ROOT/'target-draft-metadata-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n');(ROOT/'target-draft-metadata-source-excerpts.json').write_text(json.dumps(dict(qualification=review['qualification'],methods=excerpts),ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(controls=len(controls),body_median_ms=price['body_call_median_ms'],guarded_11_frame_ms=means['guarded_application_ms'],guarded_pct=means['guarded_application_ms']/window*100,conditional_request_scenarios=scenarios,new_admissions=0),ensure_ascii=False))
if __name__=='__main__':main()
