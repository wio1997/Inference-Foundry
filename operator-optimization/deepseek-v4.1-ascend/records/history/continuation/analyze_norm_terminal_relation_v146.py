"""Typed normalization/value/demand contracts; no model or device execution."""
from pathlib import Path
from collections import Counter
from itertools import product
import hashlib,json,math,statistics
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def demand(n,u,w,a,scheme):
 out=set(u);ls=[]
 for l in [1,0]:
  q=set(out);keys=set(w)
  for i in q:keys.update(range(i+1)if scheme=='CAUSAL'else range(n)if scheme=='DENSE'else[i])
  nr=q|keys;raw=nr|(set(a)if l==1 else set())
  ls.append(dict(layer=l,output_rows=sorted(out),query_rows=sorted(q),KV_rows=sorted(keys),input_norm_rows=sorted(nr),input_raw_rows=sorted(raw)))
  out=raw
 return dict(layers=sorted(ls,key=lambda x:x['layer']),total_input_norm_rows=sum(len(x['input_norm_rows'])for x in ls),initial_raw_rows=sorted(out))


def norm(v,eps=1e-6,gamma=None):
 r=1/math.sqrt(sum(x*x for x in v)/len(v)+eps);return [r*x*g for x,g in zip(v,gamma or[1]*len(v))]


def main():
 assert not(O/'REVIEW-CHECKPOINT-v146.json').exists()
 p=read(O/'norm-terminal-native-frontiers-v146.json');b=p['bindings'];parts=[];root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend'
 for rel,a,z in [('models/deepseek_v41/model.py',844,920),('models/deepseek_v41/model.py',1556,1565),('models/deepseek_v41/model.py',1592,1608),('models/deepseek_v41/dspark.py',248,309),('models/deepseek_v41/dspark.py',326,332),('models/deepseek_v41/dspark.py',403,441),('worker/model_runner_v1.py',2060,2117),('ops/layernorm.py',94,104)]:
  q=(W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend'/rel)if rel=='ops/layernorm.py'else root/rel
  parts.append(dict(**pin(q),first_line=a,last_line=z,source=q.read_text().splitlines()[a-1:z]))
 facts=[
  'Input HC Y->inputRMS Z is not MoE U:attention Post->FFNPre(current attention pre4)->RmsNormCast->MoE shared/routed merge->FFNPost. F23->next-layer F02 can compose Post/nextcollapse/RMS publication only while retaining BF16 Post/Y/Z and all current coeff/pre4/full4H residual/Engram/aux/state readers. Source adjacency/no intervening Compute is not allreader closure.',
  'Target final weighted HCcollapse consumes final FFNPost and final premix;final RMS consumes that rounded BF16 collapsed value. Neither inputRMS output nor auxiliary mean is an equal value by H5120. 24 selected eager roots positively bind both full4H MTP snapshot and later final collapse/RMS. Snapshot remains an independent fullstate publication;actual later consumed rowset is unbound.',
  'Keeping the full raw4H MTP publication does not itself force every later collapse/norm row:with actual demand U,collapse/norm can be selected while full snapshot is retained. It prevents declaring the upstream full4H state dead without closing snapshot observers. Full current baseline snapshot extent is not proof all its rows are semantically read.',
  'Auxiliary checkpoint mean reads residual entering layer,not weighted Y or Z. A joint HC/mean producer can share a chosen raw4H pass but must emit distinct rounded values and retain residual/coefficient readers. Main-context RMS follows main projection/full-feature gather;draft inputRMS follows layer HC;draft headRMS follows final raw collapse. Distinct inputs/params/positions and row domains rule out unconditional CSE.',
  'Context precompute and query preparation can overlap per required layer/cache range when actual context input is ready. Fusing delayed context/query projections competes with early bank retirement;neither substitutes main_norm(context) for input_norm(query). All layer-specific weights/gamma/epsilon/cache maps and last read/old writer/queue epochs stay explicit.',
  'Final raw draft collapsed BF16 remains optional confidence input;head norm is a different value. Native104 headRMS observations do not attest optional confidence execution/absence. Later confidence/keep cannot prune its own current decoder or metadata;next target accepted outcome and next seed are distinct generations.',
  'Norm can share a producer with valid-row/slot/metadata publication only after actual current guards are available. Equal shape,graphbucket or BOOL storage does not attest same values/epochs;9regimes1352executions remain pending. Same-slot next iteration need not have same hidden input or C2 old ring.',
  'Input-normalized Z supports selected Q plus required KV/Index/C1/C2/Weight/rawnorm readers. Selecting terminal output rows alone cannot delete inputRMS rows needed by current/future cache/state. F34 quotient needs actual accepted/rollback/oldring/halo demand and original roundedZ->FLOAT/pool/state contracts;new representation cannot infer old state from current rstd.',
  'Row-demand model is a declared two-layer value graph:rowwise HC/FFN outputs,three chosen attention windows,same supplied state-write set W at each layer,one entering-last-layer aux demand A. No dynamic routing/error/RNG/collective/EPLB/capture/all-state closure is claimed. Backward graph reachability,not selected-output cardinal alone,determines inputNorm rows.',
  'Saved cached native-API tasklinks and raw API CONNECTION_IDS family membership are different relations:24MTP hidden AGs have no cached linkedTASK,while raw families contain notification/AICPU tasks. Preserve both and exactCOMM-connection taskrows;not zero communication/cost nor unique Compute-role promotion.',
  'Classical direct statistics/matrix/value-interface work remains domain-specific. Generic physical lower bound needs nonduplicate necessary work divided by matching valid capacity upper;take max across resources permitting overlap. 76GH selected collapse/RMS private-interface union and16GH raw snapshot copy are logical chosen materializations,not HBM traffic,allocation or physical time. No native implementation or actual fusion gain.'
 ]
 write(O/'norm-terminal-source-contract-v146.json',dict(parts=parts,facts=facts,all_runtime_gates_proved=False,new_device_work=False,overall_complete=False))
 tg=Counter();dg=Counter();spans=[];taskdur=Counter();targetwork=[];targetobserved=[]
 for r in p['target_tails']:
  s={k:b[i]for k,i in r['stages'].items()};m=r['owner_rows'];g=int(s['TERMINAL_NORM']['definition']['inputShapes_text'].strip('"').split(',')[0]);t=r['cache_rows'];tg[m,g,t]+=1
  stages=[s[k]for k in ['MTP_SNAPSHOT','COLLAPSE_CAST_IN','COLLAPSE_MUL','COLLAPSE_SUM','COLLAPSE_CAST_OUT','TERMINAL_NORM']];lo=min(x['TASK']['startNs']for x in stages);hi=max(x['TASK']['endNs']for x in stages);dur=sum(x['TASK']['endNs']-x['TASK']['startNs']for x in stages);spans.append((hi-lo,dur))
  targetobserved.append(dict(key=[r['database_index'],r['forward_API_rowid']],M=m,G=g,T=t,current_six_TASK_duration_sum_ns=dur,current_selected_task_envelope_ns=hi-lo,current_snapshot_then_collapse_norm_order=all(a['TASK']['endNs']<=z['TASK']['startNs']for a,z in zip(stages,stages[1:])),raw_snapshot_collapse_norm_different_values=True,actual_fusion_critical_path_gain_ns=None))
 for r in p['draft_frontiers']:
  m=int(b[r['layers'][0]['INPUT_NORM']]['definition']['inputShapes_text'].strip('"').split(',')[0]);hrows=int(b[r['head_norm']]['definition']['inputShapes_text'].strip('"').split(',')[0]);dg[r['context_rows'],r['query_rows'],m,hrows,r['ancestry']]+=1
 for(m,g,t),cnt in sorted(tg.items()):
  h=5120;targetwork.append(dict(M=m,G=g,T=t,frontiers=cnt,H=h,input_norm_owner_Y_private_interface=4*m*h,raw_MTP_value_content=8*g*h,raw_MTP_snapshot_copy_interface=16*g*h,premix_global_FLOAT_value_content=16*g,collapse_X_FLOAT_private_interface=32*g*h,collapse_product_FLOAT_private_interface=32*g*h,collapse_sum_FLOAT_private_interface=8*g*h,collapse_BF16_private_interface=4*g*h,collapse_RMS_private_union=76*g*h,current_terminal_norm_direct_scalar_reference=4*g*h+g,current_terminal_norm_direct_rsqrt_reference=g,physical_IO_capacity_time_or_MFU=None))
 def st(v):return dict(min=min(v),median=statistics.median(v),max=max(v),count=len(v))
 write(O/'norm-terminal-observed-and-work-v146.json',dict(targets=targetobserved,target_geometry=targetwork,draft_geometry=[dict(context_rows=c,query_rows=q,input_norm_owner_rows=m,head_norm_global_rows=hr,ancestry=a,frontiers=num)for(c,q,m,hr,a),num in sorted(dg.items())],selected_target_envelope_ns=st([x[0]for x in spans]),selected_target_TASK_sum_ns=st([x[1]for x in spans]),same_device_current_task_duration_not_gain_or_request_latency=True,source_bound_MTP_COMM_records=48,MTP_hidden_without_cached_native_API_linked_TASK=24,MTP_premix_with_cached_native_API_linked_TASK=24,raw_API_connection_TASK_membership_rows=sum(len(c['raw_API_connection_TASK_rows'])for c in p['collectives']),raw_family_not_unique_Compute_role_proof=True,new_native_or_phase_promotions=0,new_device_work=False,overall_complete=False))
 cases=[];summary=Counter();n=4;h=5120
 for scheme,um,wm,am in product(['CAUSAL','DENSE','SELF'],range(16),range(16),range(16)):
  u=[i for i in range(n)if um>>i&1];w=[i for i in range(n)if wm>>i&1];a=[i for i in range(n)if am>>i&1];d=demand(n,u,w,a,scheme);nr=d['total_input_norm_rows'];summary['FULL_STATE_CASES'if wm==15 else'OTHER_STATE_CASES']+=1
  if wm==15:assert nr==8
  cases.append(dict(N=n,scheme=scheme,terminal_U=u,state_W=w,last_layer_aux_A=a,result=d,logical_Y_edge_pruning_credit=4*h*(2*n-nr),logical_Y_edge_remaining_fusion_credit=4*h*nr,logical_Y_edge_joint_union=4*h*2*n,incorrect_full_fusion_plus_pruning=4*h*2*n+4*h*(2*n-nr),all_native_or_state_gates_closed=False))
 write(O/'norm-terminal-backward-demand-cases-v146.json',dict(cases=cases,count=len(cases),summary=dict(summary),H=h,scope='Finite value-graph domainN4/twoLayers/allchosenUWAsets/3windowpatterns;independentgraph-reachabilityverifierrequired. Not all originalmodel domains/native/runtime/allreaders closed.',physical_lower_bound_or_actual_gain=False,new_device_work=False,overall_complete=False))
 nodes=['SAMPLE0','NEXT0','META0','HC0','RMS0','CACHE0','DRAFT_COLLAPSE0','HEAD_NORM0','CONF0','KEEP0','META1','VERIFY1','SAMPLE1','NEXT1'];edges=[['SAMPLE0','NEXT0'],['NEXT0','META0'],['META0','HC0'],['HC0','RMS0'],['RMS0','CACHE0'],['CACHE0','DRAFT_COLLAPSE0'],['DRAFT_COLLAPSE0','HEAD_NORM0'],['DRAFT_COLLAPSE0','CONF0'],['HEAD_NORM0','CONF0'],['CONF0','KEEP0'],['KEEP0','META1'],['META1','VERIFY1'],['VERIFY1','SAMPLE1'],['SAMPLE1','NEXT1']]
 witnesses=dict(
  same_Z_different_aux=dict(X0=[[1.],[1.],[1.],[1.]],X1=[[1.],[5.],[5.],[5.]],premix=[1.,0.,0.,0.],collapsed_Y=[1.,1.],normalized_Z=[norm([1.]),norm([1.])],aux_mean=[1.,4.],aux_reconstructible_from_Z=False),
  same_H_different_norm=dict(input=[1.,2.],terminal=[2.,1.],input_Z=norm([1.,2.]),terminal_Z=norm([2.,1.]),same_input_different_gamma=norm([1.,2.],gamma=[1.,2.]),same_input_different_epsilon=norm([1.,2.],eps=1.),unconditional_CSE=False),
  final_demand_not_input_demand=dict(N=6,ordered_head_I=[2,2,0],additional_final_J=[1,3],terminal_U=[0,1,2,3],required_cache_W=[0,1,2,3,4,5],required_input_norm=[0,1,2,3,4,5],head_row_multiset_preserved=True,all_native_reader_sets_attested=False),
  raw_snapshot_independent=dict(G=8,baseline_snapshot_raw_rows=list(range(8)),conditional_terminal_U=[0,2],conditional_collapse_norm_rows=[0,2],delete_raw_snapshot_authorized=False,full_snapshot_does_not_force_full_later_norm=True,actual_snapshot_consumed_rows=None),
  context_query_head_distinct=dict(main_context_rows=6400,input_norm_owner_rows=2,uniform_input_carrier_rows=16,head_norm_global_query_rows=10,all_H=5120,equals_by_H=False),
  epoch_mask_and_ring=dict(old=dict(request='A',generation=1,valid=[True,False],ring=2.),new=dict(request='B',generation=2,valid=[False,True],ring=10.),shape=[2],old_mask_or_ring_reusable_by_shape=False),
  acceptance_graph=dict(nodes=nodes,edges=edges,illegal_future_edge=['KEEP0','META0'],current_decoder_pruned_by_its_own_confidence=False,confidence_active_in_frozen_profile_attested=False),
  witness_scope='Constructed exact-real/set/source-domain examples only. No frozen tensor equality,optional-branch attestation,native round parity or actual gain.',new_device_work=False,overall_complete=False)
 write(O/'norm-terminal-value-row-epoch-witnesses-v146.json',witnesses)
 pairs=[['F02',x]for x in ['F23','F26','F27','F29','F30','F31','F34','F35']];contracts=[
  'Shared/routed merge->FFNPost->nextHCcollapse/RMS can share a declared producer/handoff onlyafter BF16 Merge/Post/Y/Z,pre4,full4H residual/Engram/aux observers and coefficient deadlines;not same pre-attn normalized value or additive interface caps.',
  'InputRMS and finalweightedcollapse/RMS have distinct values and rows. Preserve full4H snapshot demand independently;conditional finalU may shrink collapse/norm without deleting fullsnapshot,while input rowsfollow KV/state+query demand.',
  'Auxmean may share one raw4H HC pass but cannot reuse weightedY/Z;main-context RMS follows distinct projected/fullfeature input. Exactrow/parameter/rounding and transport/allreader gates price one chosen joint graph.',
  'Draftquery inputRMS/currentHC and contextmain/KV norms use distinct values,layerweights/eps/slots. Per-rangeoverlap orjointprojection requires bothready and extendsretirement appropriately;no H5120 CSE.',
  'Draft rawcollapse/headRMS/confidence are separate;104headRMS differ from312ownerinputRMS. Keeprawconfidence and nativeBF16 barriers;futurecutoff cannot prune its own current producer ormetadata.',
  'Normproducer/rowmask/slotmetadata may share publication after currentguard/row/bank/queue epochs;old same-shape BOOL/ring/graphbucket and future acceptedoutcome cannot supply current input contract.9/1352 remainpending.',
  'NormalizedZ/C2FLOATprojection/pairpool/FP32ring old-futurestate demands survive terminalQpruning unless allstate observers orvalidquotient closed. Samecurrentrstd doesnotrecoveroldring/halo;preserve native rounds andall-ownerrelease.',
  'Terminal orderedhead/multiset+prompt+raw readers define outputU,whereas inputNormrowsare backwardquery/KV/state/auxclosure. HeadKwork/duplicates/RNG/state remain;pruningandremaining-Y-interfacefusion useoneunion,notstandalonecaps.'
 ];ext=[['HG73-TAIL','F02'],['HG85-VERIFY-CONTEXT-DRAFT','F02']]
 write(O/'norm-terminal-composition-v146.json',dict(clauses=[dict(id='REL146-'+str(i+1).zfill(2),a=a,b=b,contract=c)for i,((a,b),c)in enumerate(zip(pairs,contracts))],new_pairs=pairs,group_extensions=ext,new_groups=0,all_runtime_gates_proved=False,whole_pair_or_all_subsets_closed=False,speedup_additive=False,overall_complete=False))
 names=['norm-terminal-native-frontiers-v146.json','hc-moe-native-frontiers-v141.json','hc-draft-native-frontiers-v142.json','hc-draft-source-contract-v142.json','hc-norm-attention-native-frontiers-v145.json','hc-norm-attention-observed-and-work-v145.json','hc-norm-index-state-work-and-lifetimes-v144.json','terminal-layer-demand-cut-v86.json','DSpark-tail-reader-closure-v85.json','aux-sp-causality-and-variant-graph-v131.json','verify-metadata-value-and-epoch-witnesses-v129.json','sampling-visibility-proof-and-boundaries-v117.json','candidate-relation-clauses-v145.json','norm-terminal-postcheck-placeholder','hc-norm-attention-postcheck-result-v145p.json']
 names.remove('norm-terminal-postcheck-placeholder')
 write(O/'norm-terminal-inputs-v146.json',dict(inputs=[pin(O/x)for x in names],questions=['Can terminal output selection prune inputRMS in the presence of KV/state/rawsnapshot/aux readers?','Which RMS values/rowdomains andtyped intermediates areactually distinct?','Can interlayer/aux/draft/metadata/state composition share a materialization without reusing wrong values/epochs or doublecounting?'],profile_first=True,target_source_scope='Only direct HC/norm/tail/aux/draft-head/context/metadata/layernorm value andreader sites;no frameworksurvey',new_device_work=False,overall_complete=False))
 print(json.dumps(dict(status='SCOPED_OPERATOR_DEMAND_REVIEW',native_unique_TASKs=p['unique_TASK_bindings'],new_source_CID_bindings=248,targets=24,draft_head_norms=104,aux=72,interlayer_bridges=936,demand_cases=len(cases),full_state_cases=summary['FULL_STATE_CASES'],new_pairs=len(pairs),new_device_work=False,overall_complete=False)))


if __name__=='__main__':main()
