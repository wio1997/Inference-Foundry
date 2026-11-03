from copy import deepcopy
from datetime import datetime,timezone,timedelta
import json,hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write,ctl

def main():
 assert not(O/'REVIEW-CHECKPOINT-v24.json').exists()
 assert not(R/'candidates/F34.json').exists()
 now=datetime.now(timezone.utc);stamp=now.isoformat()
 labels=['compressor-boundaries166-v24','C2-row-domains166-v24','C2-source166-v24']
 hashes=['b5b12c5860c862c3c824bb2f4e3e805b1fd0fdebf01c3a895beed3d5bace13b4','9aaf6179fe938556de2de75b6df23f8003b589da1bada4aeda72adf446d3dc0b','421aae17d288d7ddaf0630ae625c0454b6db2522806498ec1cabb0e6fcad42d8']
 data={};refs=[]
 for label,sha in zip(labels,hashes):
  r=read(N/(label+'.json'));assert r['returncode']==0 and not r['stderr'];z=json.loads(r['stdout']);assert z['result_sha256']==sha;data[label]=z;refs.append(dict(receipt=str(N/(label+'.json')),path=z['full_result'],sha256=sha))
 profile=data[labels[0]]['summary'];schedule=data[labels[1]]['summary'];source=data[labels[2]]['result'];proof=read(O/'C2-ring-observability-integer-review-v24.json')
 assert profile['counts']==dict(graph_pool=240,graph_ring_update=240,selected_C2_blocks=240)
 assert profile['pair_identification_errors']==[] and profile['pool_update_norm_same_TASK_stream']=={'True':240}
 assert profile['intervening_pool_to_norm_compute']=={'_cache_update_kernel':240}
 assert schedule['exact_bound_graphs']==64 and schedule['phase_unbound_graphs']==16
 assert schedule['conditional_completed_pair_endpoint_rows']==2016 and schedule['recorded_physical_C2_norm_rows']==4608
 assert proof['integer_cases']==12804 and proof['finite_abstract_updates']==59
 native_pools=[r for r in data[labels[0]]['M08_role_counts']if r['origin']=='DIRECT'and r['signature']['opType_text']=='_pool_kernel']
 assert sum(r['executions']for r in native_pools)==72
 sourcefiles=[dict(path=f['path'],sha256=f['sha256'],functions=[dict(name=q['name'],start=q['start'],end=q['end'])for q in f['functions']],ranges=f['exact_row_contract_ranges'])for f in source['files']]
 findings=[
  dict(id='C24-1',grade='MEASURED',finding='All80 current166 graph executions contain exactly one C2 pool/update pair in each structural source block2/8/14:240pairs. They all follow pool->ring_update->pooled512norm on one TASK stream. The only intervening compute between pool and norm is ring_update.192C2blocks are decode-compatible,48 remain phase-unbound.',evidence=refs[0]),
  dict(id='C24-2',grade='MEASURED',finding='Pool->update and update->norm gaps are20ns median; pool->norm median5360ns includes the intervening update. This is recorded ordering,not a necessary dependency,causal critical-path cost or removable time.',evidence=refs[0]),
  dict(id='C24-3',grade='MEASURED',finding='Current native C2 pool/norm callers retain72executions over physical global-cache row regimes6400/6528/6668,24each. Current C1 native norm24 is separate. Do not mix these current166 records with old archived169 C2 scopes96/C1scopes32.',evidence=[refs[0],'generic-helper-semantic-review-v18.json']),
  dict(id='C24-4',grade='DOCUMENTED',finding='Pinned pool_projected passes a private FP32 state cache and separate token-aligned BF16 pooled output. Pool reads old ring,update reads FP32 kv/score+metadata and writes ring,norm reads pooled output. Empty helper launch bounds still zero output and return to the caller norm;skip metadata sets used_len0 rather than proving all surrounding work dead.',evidence=refs[2]),
  dict(id='C24-5',grade='DERIVED',finding='Under disjoint storage and listed-effect contracts,pool->update WAR andpool->norm RAW are necessary,while the intermediate update->norm host-order edge is not. A fork/deferred-update analysis must preserve update completion before future ring readers and before kv/score/metadata reuse. It cannot remove the old-read-before-write constraint.',evidence=['C2-ring-observability-integer-review-v24.json',refs[2]]),
  dict(id='C24-6',grade='DERIVED',finding='For TOKEN_ALIGNED=True,RATIO=2,CACHE_SIZE=32,write start=2q+r. Only local_group0 withr=1 can read history,and its sole old position is2q. Therefore all effective historical ring reads use even absolute positions/even slots. Preserving every even-slot FP32 write and generation preserves future current-helper history values,including rollback,in the stated abstract state domain. Other consumers/state ABI/native binding remain obligations.',evidence=['C2-ring-observability-integer-review-v24.json',refs[2]]),
  dict(id='C24-7',grade='DERIVED',finding='New F34 studies observable-state pruning: omit odd-position maintenance with existing layout,or separately study a16-slot representation. It does not permit retaining only the last token:after processingpositions0..5,a rollback nextstart1needs oldposition0,not latestevenposition4. A consumer of an odd ring entry also distinguishes full andfiltered state,so full-buffer equality and arbitrary-reader domains exclude this rewrite.',evidence='F34-C2-ring-observability-candidate-v24.json'),
  dict(id='C24-8',grade='DOCUMENTED',finding='The same _pool_kernel already has a nonTOKEN_ALIGNED fused RMS branch. The current wrapper selects TOKEN_ALIGNED=True,passes out as the ignored norm-weight pointer andeps0,and uses token-completion row layout. Flipping that flag alone would expose placeholder weights/epsilon andcompact row addressing;it is not a drop-in existing pool+nativeRMS implementation. Native RMS arithmetic equivalence remains open.',evidence=refs[2]),
  dict(id='C24-9',grade='DERIVED',finding='Exact graph-root metadata binds64graphs:16 with2requests x6scheduledtokens and48 with4x6. If each corresponding C2 device chunk uses those contiguous n positions,there are2016 pair endpoints across192sourceblocks versus4608physical normrows,fraction7/16. This is conditional schedule arithmetic across rank/graph/source records,not measured valid device rows or a compute/time saving. Invalid pages/slots/skip andpostcorrection metadata remain unobserved.',evidence=refs[1]),
  dict(id='C24-10',grade='DERIVED',finding='Under the same conditional six-token chunks,the full ring tail update covers4032row records;even-only2016. At4096logical FP32 kv+score bytes per row,the interface models are16515072 vs8257536bytes. The native[17306,32,1024]FP32 array model is2268332032bytes and16slots1134166016bytes. Neither model establishes HBM traffic,resident memory,allocator savings,necessary I/O or speedup.',evidence='C2-ring-observability-integer-review-v24.json'),
  dict(id='C24-11',grade='UNKNOWN',finding='All ring consumers/aliases andstate ABI,page/request write ownership,actual metadata generations,packed/legacy paths,allocator padding and native compilation binding remain open. No NPU implementation,legal native fusion,capacity ceiling or measured gain is claimed.',evidence='M08-C2-state-condition-ledger-v24.json')]
 write(O/'C2-state-opportunity-contract-v24.json',dict(generated_utc=stamp,status='FINITE_SOURCE_STATE_AND_CURRENT_PROFILE_DOMAINS_REVIEWED',findings=findings,source_files=sourcefiles,legal_native_fusion_proved=False,measured_gain=None,whole_module_closed=False,overall_complete=False))

 branch_specs=[
 ('C2-RATIO','ratio1 / ratio2 / foreignratio','Current ratio2 token-aligned helper domain;C1 has no gate/ring path.','F34 is not an all-ratios or C1 rewrite.'),
 ('C2-EMPTY','tokens/batches/max_query_len zero','Output zeroing precedes early return;caller still runs norm.','No blanket removal of surrounding output/norm/cache semantics.'),
 ('C2-PAGE','page<=0 / page>=numblocks / valid','TOKEN_ALIGNED pool andPROTECT_NULL update guard page bounds.','Retain null/padding semantics and actual ownership;request index differs from physicalpage.'),
 ('C2-USED','live requests/actual tokens/query lengths/skip','Nonpacked builder clampsusedlength toactualtokens,zerosdeadrequests andskip;start derivesseq_len-query_len.','Host schedule is not the device used_len or actual slot payload.'),
 ('C2-START-PARITY','start even / odd','Even start has no old ring read;odd start can read one even predecessor.','Retain same FP32 kv/score and generation on that predecessor.'),
 ('C2-POOL-ZERO','complete endpoint / incomplete or invalidrow','Zero all token-aligned rows then only write complete endpoints.','Skip/prune only after all unproduced-row consumers andzero behavior are bound.'),
 ('C2-NORM','TOKEN_ALIGNED / nonTOKEN_ALIGNED','Current external nativeRMS;other branch contains fused math butdifferent pointer/eps/layout contract.','Flag-only change excluded;retain BF16pool rounding,weights,epsilon andnative accumulation contract.'),
 ('C2-TAIL','used<32 / used>=32','Update stores lastmin(used,32)FP32kv+score rows.','Keep full relevant even suffix;do not replace with latest token under rollback.'),
 ('C2-FP32','projection/ring FP32 / otherdtype','Current helper explicitly checksFP32 and matching[rows,width];outBF16,widthpower2.','F34 does not authorize dtype downcast or alteredpool arithmetic.'),
 ('C2-LAYOUT','contiguous / columnview/compact','Helper requirescontiguous projections,state,metadata andout;existingfullring32-row stride.','ExistingF17naivepackedcolumnview exclusion persists;16slot layout is another domain.'),
 ('C2-P-U-WAR','old-state read vsretainedeven-slot overwrite','P->U stillrequired;oldslot0 may beoverwritten bynewtoken32.','Odd-write pruning doesnot eliminate the current dependency witness.'),
 ('C2-U-N-FORK','disjoint ring andpooled/normstorage / aliases','Abstract update/norm read/write sets commute when disjoint.','Bind actualaliases,purity and asynchronous lifetime before native rescheduling.'),
 ('C2-DEFERRED-JOIN','same-callreturn / future reader / inputreuse','Deferred update must finish beforefuture ring reads andbeforekv/score/metadatareuse.','Do not equate Python enqueue return orpool completion withupdate completion.'),
 ('C2-ROLLBACK','append / accepted-prefix rewind','Every even-position generation usedbyfuturepool must matchbaseline.','Singleton latest-token retention isexcluded bynextstart1after0..5 witness.'),
 ('C2-STATE-OBSERVABILITY','onlycurrentpool reads / foreignreader/fullbytes ABI','Even-slot invariant givesobservational equivalence onlywithrestrictedconsumers.','Anodd-slot foreignreader distinguishescandidate;allconsumerclosure remainsopen.'),
 ('C2-WRITE-OWNERSHIP','distinct pages / duplicate orrecycled page','Allsame-address write ordering andgeneration rulesmustpersist.','Do not inferdisjointness fromdifferentrequest IDs;finite sequential model isnot a native race proof.'),
 ('C2-CACHE-LAYOUT','existing32slot / compact16slot / packedbackend','Compact model changespagebytes,addressformula,cache spec andcapture addresses.','No actualallocation savings without allocator/page-padding/readers andpacked-interface audit.'),
 ('C2-INDEX-LONG','pooledlatent -> indexkey /longKV','Keeporiginal pre-longRoPE indexinput,pairedINT8key+FP16scale andlongKVpublication.','Different slot/position coordinates andcrosslayerconsumers persist;reuseF19contract.'),
 ('C2-CAPTURE','192boundsourceblocks /48phaseunbound','Currentprofileboundarycensus coversall240;conditionalrowmodelusesonly64boundgraphs.','No phase invention orgraph-address/generation closure fromcounts.'),
 ('C2-ROW-ACCOUNTING','scheduled /complete /validpublished /physical','Understateddevicecontract,2016pairendpoints/4608physicalnormrows;slotinvalidity mayremove endpoints.','7/16 isnot measuredvalidrows,compute savings,HBMtraffic orrequestgain.')]
 branches=[dict(id=i,condition=c,current_source_or_profile=s,remaining_obligation=g)for i,c,s,g in branch_specs]
 write(O/'M08-C2-state-condition-ledger-v24.json',dict(generated_utc=stamp,branches=branches,scope='Finite currentC2state/pool/norm/row subtree;not allM08/M24 modules.',all_module_branches_closed=False,whole_module_closed=False,overall_complete=False))
 reorg=dict(generated_utc=stamp,current_graph=['zero_pooled','pool_old_ring','ring_update','pooled_norm','index_key_then_longKV'],
  observed_current_pairs=240,baseline_happens_before=[['zero_pooled','pool_old_ring'],['pool_old_ring','ring_update'],['ring_update','pooled_norm']],
  conditional_semantic_edges=[dict(a='zero_pooled',b='pool_writes_completed_rows',kind='WAW'),dict(a='pool_old_ring_reads',b='ring_update_same_slot',kind='WAR'),dict(a='pooled_BF16',b='norm',kind='RAW'),dict(a='ring_update',b='future_ring_reader',kind='RAW_AND_GENERATION')],
  variants=[dict(id='F18-V24-A',graph='Pool/norm value path andringmaintenance fork afteroldringread completion',status='CONDITIONAL_DISJOINT_EFFECTS_DOMAIN'),dict(id='F18-V24-B',graph='Pool+norm semanticboundary withtoken-aligned zero/completeoutput andunchangedRMScontract',status='CONDITIONAL;NOT_A_NEW_MISSING_FUSED_MATH_PRIMITIVE'),dict(id='F18-V24-C',graph='Flip currentTOKEN_ALIGNED flagonly',status='EXCLUDED_IN_UNCHANGED_ARGUMENT_AND_LAYOUT_DOMAIN',reason='Wrapperpassesoutasnormweight,eps0,andtoken-aligned metadata;nonalignedbranchusescompactrows.')],
  limits=['Retain original BF16pool boundary and full-row RMS arithmetic.','No current barrier is removed or native scheduleimplemented.','Existing kernel nonaligned fused-math source isnot proof ofcurrentloaded path/equivalence.','F34even-state pruning cannot deleteold-read-before-write WAR.'],overall_complete=False)
 write(O/'F18-pool-update-norm-domains-v24.json',reorg)
 f34=dict(id='F34',name='C2 ring 状态的可观测性裁剪、稀疏维护与表示重组',profile_review_state='NEW_CONDITIONAL_SOURCE_STATE_OPPORTUNITY;NATIVE_AND_ALL_CONSUMER_BINDING_OPEN',
  original_template_graph_id='ATTN_C2_SOURCE',original_node_mapping=dict(ring_pool='Source-refined stateful compound:old-ringpool + tail-stateupdate;do not treatrefinement asalreadydistinctG1nodes.'),
  source_semantics='TOKEN_ALIGNED=True,RATIO=2,CACHE_SIZE=32. Effective historical reads useeven positions/slots;preserveallvalid even-slot FP32values andgenerations.',
  candidate_graph=dict(nodes=['C2_FP32_kv_score','pool_reads_old_even_ring','token_aligned_BF16_pooled','norm_and_dual_publication','ring_maintenance'],
   retained_edges=[dict(a='C2_FP32_kv_score',b='pool_reads_old_even_ring',kind='RAW'),dict(a='pool_reads_old_even_ring',b='ring_maintenance_same_slot',kind='WAR'),dict(a='ring_maintenance',b='future_pool_read',kind='PERSISTENT_STATE_GENERATION'),dict(a='token_aligned_BF16_pooled',b='norm_and_dual_publication',kind='RAW')]),
  variants=[dict(id='F34-A',change='Keep32-slot physical layout;study suppressingodd-position maintenance',status='CONDITIONAL',requires='Onlyspecifiedeven-observing ringconsumers;sameevenwrite values/order/pages/generations;otherodd-byteABIunobservable.'),dict(id='F34-B',change='Study16-slot representationforevenstate',status='CONDITIONAL_SEPARATE_LAYOUT_DOMAIN',requires='Remapallreaders/writers,cache-spec/pagepadding,allocation/packedinterfaces,checkpoint/stateABI andgraphaddresses.'),dict(id='F34-C',change='Retainonlylatesttoken/latest-even token',status='EXCLUDED_IN_ROLLBACK_DOMAIN',reason='Afterprocessingpositions0..5,nextstart1readsoldposition0,notlatest-even4.'),dict(id='F34-D',change='Applyfilterwhileallowingfull-bufferobservability orarbitrary foreignreader',status='EXCLUDED_IN_THAT_DOMAIN',reason='Odd-slot readerdistinguishes newvsoldstate bytes;RATIO!=2notcovered.')],
  proof_scope='Naturalinteger/read-set argument plusfinite12804addresscases/9abstractsequences/59updates;not a nativecorrectnesstest orformalchecker.',
  relation_to_existing='F17optimizesprojection production,F18pool/normvalueboundary,F19dualpublication.F34changespersistentstateobservablecontent/representationacrossiterations;donotdoublecounttheirboundaries.',
  dependencies=['Allringconsumers/alias/stateABI','Same-address writeownership andpagegeneration','Metadata/ready/inputlifetimes','CurrentTriton/nativecompileandpackedbranch binding','Allocatorpaddingforcompactvariant'],
  evidence_files=['C2-state-opportunity-contract-v24.json','C2-ring-observability-integer-review-v24.json','M08-C2-state-condition-ledger-v24.json'],
  legal_fusion_proved=False,candidate_closed=False,critical_path_gain_us=None,critical_path_gain_status='UNKNOWN_NOT_LOGICAL_BYTE_RATIO',implementation=False,new_performance_result=False,overall_complete=False)
 write(O/'F34-C2-ring-observability-candidate-v24.json',f34);write(R/'candidates/F34.json',f34)

 model=deepcopy(read(O/'candidate-profile-ledger-v23.json'));assert len(model['candidates'])==33 and not any(c['id']=='F34'for c in model['candidates'])
 for c in model['candidates']:
  if c['id']in ['F17','F18','F19']:
   c['current_C2_state_v24']='C2-state-opportunity-contract-v24.json;240currentgraphC2boundarypairs,72nativeC2callers;old169records kept separate.'
   c['state_maintenance_relation_v24']='F34observational statepruning;F18forkrequirespoolWAR andfuture-state/input-lifetimejoin;no additivegain.'
  if c['id']=='F18':c['new_condition_domains_v24']='F18-pool-update-norm-domains-v24.json;existingnonalignedfusedmathisnotflag-onlyreplacement;20finiteC2branches.'
  if c['id']in ['F18','F19']:c['row_count_scope_v24']='64exactgraph-root schedules imply2016pairendpoints/4608physicalnormrows onlyundercontiguousused_lencontract;devicevalidpublishedrowsunmeasured.'
 model['candidates'].append(f34);model.update(generated_utc=stamp,parent_ledger='candidate-profile-ledger-v23.json')
 for k in ['count','model_candidate_count','existing_model_candidates']:
  if k in model:model[k]=34
 write(O/'candidate-profile-ledger-v24.json',model)
 framework=deepcopy(read(O/'framework-candidate-ledger-v23.json'));framework.update(generated_utc=stamp,parent_ledger='framework-candidate-ledger-v23.json');write(O/'framework-candidate-ledger-v24.json',framework)
 mods=deepcopy(read(O/'module-profile-ledger-v23.json'));matrix=deepcopy(read(O/'module-phase-branch-matrix-v23.json'))
 notes={'M02':'F34compactstate isonlyalogicalarray/cachespecreorganizationdomain;actualpagepadding/sharedallocationandgenerationreconciliationremainopen.','M08':'All240currentC2graph pool/update/normboundariesand72nativeC2poolsreviewed;newF34even-state invariantand20finitebranches;native/allconsumerclosureopen.','M09':'Conditionalcompletedpair-row pruningpreserves indexkey/FP16scale andlongKVversions;host2016endpointsarenotmeasuredvalidcachewrites.','M22':'CapturedTOKEN_ALIGNEDpathkeepsseparateRMS;nonalignedfusedbranchisnotdrop-in.Captureaddresses/deferredstatejoinsmustpersist.','M24':'Even-slot stateinvariantcoversarbitraryfuturecurrent-helperstartsintheabstractdomain;latest-onlyretentionfailsunderrollback.Allreaders/page/alias/generationconstraintsremain.'}
 for m in mods['modules']:
  m['phase_branch_record']='module-phase-branch-matrix-v24.json#'+m['id']
  if m['id']in notes:m['review_v24']=dict(finding=notes[m['id']],evidence=['C2-state-opportunity-contract-v24.json','M08-C2-state-condition-ledger-v24.json','F34-C2-ring-observability-candidate-v24.json'],whole_module_closed=False)
  if m['id']in ['M02','M08','M22','M24']and'F34'not in m['candidate_links']:m['candidate_links'].append('F34')
 for m in matrix['modules']:
  if m['id']in notes:m['branches'].append(dict(branch='CurrentC2stateobservability/pool-update-norm andconditionalrowdomain',state='FINITE_SUBTREE_REVIEWED;ALL_CONSUMER_AND_WHOLE_MODULE_OPEN',evidence=notes[m['id']],gap='Keepactualmetadata/readers/native/allocator/stateABIandothermoduleconditionsopen.'))
 mods.update(generated_utc=stamp,parent_ledger='module-profile-ledger-v23.json',existing_model_candidates=34);matrix.update(generated_utc=stamp,parent_matrix='module-phase-branch-matrix-v23.json');write(O/'module-profile-ledger-v24.json',mods);write(O/'module-phase-branch-matrix-v24.json',matrix)
 remaining=deepcopy(read(O/'module-coverage-gaps-v23.json')['remaining'])
 remaining.insert(1,'NewF34:bindallcurrentC2ringreaders/stateABI andpage/writeownership beforeupgradingeven-stateobservationalpruning;compact16slotvariantaddscachespec/padding/allocation/packed/capturecontracts. No arbitraryfullstate-byte equivalence.')
 remaining.insert(2,'Current240C2pool/update/normpathandconditional2016/4608endpointaccountingreviewed;notmeasuredvaliddevice rows. Ifrowpruningconclusionneedsit,bindCPglobalmetadataused_len/live/slot/skipgenerationusingexistingprofile-firsthandoff evidencebeforepointsource.')
 remaining[-1]='F33/H07andremainingcompressor/ring/rollback/allotherboundaryconsumersremainopen;v24addsF34andfiniteC2stateproofbutnotwholemoduleclosure.Preserve3136candidate-setresource rows/849missingPMU/16unboundgraphs/26theoryINCOMPLETE;do not inferpeak/MFU/gain.'
 write(O/'module-coverage-gaps-v24.json',dict(generated_utc=stamp,remaining=remaining,whole_modules_closed=0,overall_complete=False))
 summary=dict(generated_utc=stamp,status='PROGRESS',previous_turn_classification='PROGRESS:v22boundarycontracts andv23attributeevidencequality',new_main_candidate='F34',model_candidates=34,framework_candidates=7,current_graph_C2_pairs=240,decode_compatible_C2_pairs=192,phase_unbound_C2_pairs=48,current_native_C2_pool_executions=72,conditional_pair_endpoint_rows=2016,physical_norm_rows_in_bound_graphs=4608,conditional_endpoint_fraction=0.4375,condition_branches=len(branches),integer_cases=12804,abstract_updates=59,point_source_files=3,new_device_or_service_work=False,all_jobs_terminal=True,legal_native_fusion_proved=False,measured_gain=None,whole_modules_closed=0,overall_complete=False,evidence_refs=refs,remaining=remaining)
 write(O/'C2-state-summary-v24.json',summary)
 gate=read(O/'PROFILE-CODE-QUESTION-GATE-v24.json');gate.update(answered_by=['C2-state-opportunity-contract-v24.json','F18-pool-update-norm-domains-v24.json','F34-C2-ring-observability-candidate-v24.json'],closed_scope='NamedC2source effects,ratio2historyreadset,nonalignedfused-branch interface andconditional schedulearithmetic.',not_closed='Historicaldevice metadata/aliases/allstateconsumers/nativecompile/physicalgain.',overall_complete=False);write(O/'PROFILE-CODE-QUESTION-GATE-v24.json',gate)
 report='''本轮新增一个独立机会F34：C2 ring状态的可观测性裁剪、稀疏维护与表示重组。它来自当前166 profile提出的状态边界问题及三文件定点源码，不是重复登记F17投影融合、F18 pool/norm或F19双cache发布。整体仍未完成，没有实现内核、运行NPU或改变服务。

当前80个图中的source层块2/8/14合计240次C2 pool/update/norm边界已全部核对：同一TASK stream，pool到norm之间唯一计算是ring更新。两侧间隔中位数均20ns，pool到norm中位数5360ns包含这次更新；不能把它当可省时延。192个C2块decode-compatible，48个阶段仍未绑定。当前原生C2 pool共72次，global-cache行数6400/6528/6668各24次；这些与旧169的96个C2采样分开。

定点源码将两种依赖区分开：pool可能读取旧ring，所以同槽更新必须在旧值读取完成后；norm读取的是独立pooled BF16输出，ring更新读取FP32 kv/score并写ring。在无别名且副作用仅限这些集合的合同下，更新不必夹在pool与norm之间。可以登记值路径与状态维护分叉、pool/norm边界重组等条件域；任何延后更新都必须在未来ring读者及kv/score/metadata复用前完成。源码同流次序是当前保护方式，不是所有新排程的必要全局屏障。

新F34的整数读集合结论是：当前TOKEN_ALIGNED=True、RATIO=2、CACHE_SIZE=32。把start写成2q+r，后续完整组无需历史；首组仅在r=1时读取旧token2q。因此历史值只从偶数token位置、偶数ring槽取出。若初态的全部偶数槽相同，且保留所有偶数位置的FP32写值、顺序、页和代际，则每次更新后这一不变量仍成立，未来同一helper的pool会看到相同历史值，接受回滚也不破坏这个整数性质。

这支持两个不同条件变体：保留32槽布局而研究省去奇数位置维护；或另行研究16槽表示并相应审查全部读写、缓存规格、页padding、分配、packed backend及固定graph地址。前提是没有其他模型消费者或状态ABI观察奇数槽。一个读奇数槽的消费者会直接区分新旧状态，所以没有宣称完整state buffer逐字节相等。该结论也未推广到任意ratio或legacy接口。F34并未消除pool旧读→ring新写的WAR：原有token0/32同槽冲突仍发生在必须保留的偶数槽。

不能进一步只保留“最后一个token”或“最后一个偶数token”：处理positions0..5后，若接受回滚使下一chunk从1开始，pool需要旧position0，而最后偶数token是4。此次12,804个整数地址用例及9组/59次抽象状态更新验证了声明模型的性质和反例；自然数论证限定于所列源码语义，有限检查不是形式化验证、已加载二进制证明或native正确性测试。

还确认了一个既有能力与排除域：_pool_kernel的非TOKEN_ALIGNED分支已经包含pool后BF16转换和RMS数学。但当前wrapper选择TOKEN_ALIGNED=True，norm权重参数位置传的是被忽略的out，epsilon传0，metadata采用token对齐行。只翻转该常量会启用不适配的权重/epsilon和compact行寻址，不能直接作为当前native RMS的替代。候选仍需保留实际权重、epsilon、池化BF16舍入、全行RMS和native数值合同。

profile已有精确graph-root连接，可把64图分为16个B2×6scheduled tokens与48个B4×6。若对应device used_len确为这些连续位置，每个请求有3个完成pair，192个source块共2,016个pair-endpoint行，对比4,608个physical norm行为7/16；这不是固定“省一半”。它只是跨rank/graph/source记录的条件计数，不是已测有效cache行或计算收益。device actual-token截断、skip、null/invalid page、slot和metadata代际仍须绑定，不能从host schedule直接补出。

同一条件下ring tail写行会计为4,032行，偶数位置2,016行；每行两个512维FP32值，逻辑写入模型16,515,072与8,257,536字节。原生[17306,32,1024]FP32数组的逻辑大小2,268,332,032字节，16槽模型1,134,166,016字节。它们不是实际HBM流量、resident显存、可释放分配或时延收益；统一cache page padding等因素可能使逻辑缩小无法兑现为分配节省。

已新增20项C2有限条件、F18的三种重组/排除域及F34候选图，并将主台账扩至34项模型候选、7项框架候选。所有ring读者、状态ABI、并发写所有权、真实metadata、native编译及allocator等缺口仍明确记录，26整模块均未宣称闭合。此前33项计数不再作为最新总数；旧报告统计保留其历史范围。Goal保持active，最后健康观测仍是旧v12post。
'''
 (O/'PROFILE-FINDINGS-v24.md').write_text(report)
 pub=R/'publication-draft/FUSION-INVENTORY.md';old=pub.read_text();(O/'FUSION-INVENTORY-before-v24.md').write_text(old)
 assert '| F32 |'in old and '| F33 |'not in old and '| F34 |'not in old
 note='\n当前持续分析台账（v24）为 **34项模型候选、7项框架候选**，完整范围仍未闭合。以下早期32项及G1/G2统计保留历史口径；新增F33/F34见表。最新条件、证据及未知项以[当前台账]('+str(O/'candidate-profile-ledger-v24.json')+')和[本轮记录]('+str(O/'PROFILE-FINDINGS-v24.md')+')为准。\n'
 lines=old.splitlines();out=[]
 for i,line in enumerate(lines):
  out.append(line)
  if i==0:out.append(note)
  if line.startswith('| F32 |'):
   out.append('| F33 | C1 compressor与SWA KV的同域投影、联合分块及前移通信 | 条件 | v16补充：source层20两路BF16投影、独立512维norm；保留布局、消费者、ready和数值合同，无收益证明。 |')
   out.append('| F34 | C2 ring状态的可观测性裁剪、稀疏维护与表示重组 | 条件 | v24补充：ratio2历史有效读仅偶数槽；可研究奇数位置维护裁剪/16槽表示，必须闭合全部状态读者、ABI、页/代际/布局与native合同。回滚下只留最后token不成立，无分配或时延收益证明。 |')
 pub.write_text('\n'.join(out)+'\n')
 action=('继续Loop004，v24实质进展并新增F34，主台账34model/7framework。先读C2-state-summary-v24.json、C2-state-opportunity-contract-v24.json、F34-C2-ring-observability-candidate-v24.json、M08-C2-state-condition-ledger-v24.json。'
  '当前80图source2/8/14全部240C2pool/update/norm同stream，pool->norm之间只有update；192decode-compatible/48unbound，侧gap中位20ns，pool->norm5360ns不是收益。nativeC2pool72 M6400/6528/6668各24与old16996分开。'
  '源码3文件SHA：compressor31516e5ba234e34fa01e73e864fbc88bd6473d4a5274a1d43da8a79bab3cf494；triton1c26f996fa91bcd3ae497e1ae33e88a2e0bf38c2e734f8af85baee06492a87ee；dsa3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd。'
  '新F34在currentTOKEN_ALIGNED/RATIO2/CACHE32抽象语义下：s=2q+r，仅首partialgroup r1读旧2q，所以只读偶数位置/偶数槽。保持全部偶数FP32值/顺序/页/代际不变量可研究奇数写裁剪；16槽是另一cache-layout域。12804整数cases/9seq59updates只是有限模型佐证，不是native/形式化证明。其他reader/stateABI若读奇数会区分，必须闭合；ratio通用不涵盖。只留last-even4不能应付0..5后nextstart1需position0，接受回滚不允许singleton裁剪。WAR仍在even0/token32上，F34不解除pool旧读->新写。'
  'F18新fork域：P->U WAR和P->N RAW，disjoint U/N可不必串行；任何延后U要join未来ring读者以及K/S/metadata复用。当前wrapper依次P,U然后callerN，不能直接拿enqueue当ready。_pool_kernel已有非TOKENaligned fusedRMS，但当前out占normweight/eps0占参且compact行不同，只flip常量排除；不是缺失fused数学，也非nativeRMS替代。'
  '64exactgraphroots=16B2*6+48B4*6；在used_len=scheduled连续chunk合同下endpoint2016/physical4608=7/16，仅conditional；startparity128even96odd请求记录，post-correction/device used/slots/skip未实测。完整cache/source坐标和key+scale/long版本依旧live。'
  'F34后续先profile/state-object已保存证据查allringreader/alias/stateABI，必要时只点DeepseekV41CacheLayer和CircularBufferSpec/packed替换的具体pagepadding/ownership调用；不要扫整库或做实现。若只为定性已有conditional候选，不强迫先实现。可转H07/F33等剩余profile缺口；保持完整原目标。'
  '逻辑环tail写模型16,515,072->8,257,536B，recorded17306x32x1024FP32数组2,268,332,032B/16slot模型1,134,166,016B，不是HBM/resident/可释放显存/收益/必要IO。全局资源3136candidate/849missingPMU/16unbound/26theoryINCOMPLETE不因边界候选就擅自消除。'
  '三个server PMU结果：current-compressor-state-and-pool-norm-boundaries-v24 SHAb5b12c5860c862c3c824bb2f4e3e805b1fd0fdebf01c3a895beed3d5bace13b4；C2-exact-graph-schedule-row-domain-v24 SHA9aaf6179fe938556de2de75b6df23f8003b589da1bada4aeda72adf446d3dc0b；C2-pool-ring-norm-and-row-source-contract-v24 SHA421aae17d288d7ddaf0630ae625c0454b6db2522806498ec1cabb0e6fcad42d8。'
  'R/candidates/F34.json和publicationdraft当前34说明已更新，旧32/G1stats明确历史。Goalactive/0wholemodulesclosed。全部3remoteCPU取证+1localabstractreview terminalpass，无modelimport/NPU/服务变更；lasthealthv12post，若需补采仍先freshhealth/ownership/frozenworkload/clone/restore。')
 scripts=['query_compressor_state_boundaries_v24_166.py','query_C2_schedule_row_domains_v24_166.py','read_C2_state_and_row_contract_v24_166.py']
 descs=['All240currentgraphC2pool/update/norm boundaries and56M08resource regimes/native leaves reviewed without alias/phase invention.','Exact64graph-root schedules giveconditional2016pairendpoints/4608physicalrows;actualdevicepayload explicitly unknown.','Only3pinned C2sourcefiles/namedpool/update/helper functions androwpredicates read;nonalignedfusedbranch/interface separated.']
 jobs=[('profile'if i<2 else'review',descs[i],[N/(labels[i]+'.json'),N/scripts[i]])for i in range(3)]
 jobs+=[('review','12804integer read-set cases/9abstract sequences59updates androllback/odd-reader exclusions reviewed;no kernel/device implementation.',[N/'review_C2_ring_observability_v24.py',O/'C2-ring-observability-integer-review-v24.json',N/'zcode-C2-ring-proof-v24-output-20261031.json']),('review','Main reviewed F34newstate-observability candidate,F18conditional fork/nonaligned exclusion,20branches and34candidate inventory with fullscope unknowns retained.',[O/'C2-state-opportunity-contract-v24.json',O/'M08-C2-state-condition-ledger-v24.json',O/'F18-pool-update-norm-domains-v24.json',O/'F34-C2-ring-observability-candidate-v24.json',R/'candidates/F34.json',pub,O/'PROFILE-FINDINGS-v24.md'])]
 runs=[]
 for i,(kind,desc,arts)in enumerate(jobs):
  rid='run-'+(now-timedelta(seconds=len(jobs)-i)).strftime('%Y%m%dT%H%M%SZ');run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90-166','--kind',kind,'--command','Preparedv24 retrospective registration: '+desc,'--code-ref','Saved166profile/3exactC2sourcefiles/abstractintegerreview;no targetimplementation/NPU/servicework','--working-directory',str(N));args=['finish-run','--run-dir',run,'--status','pass','--correctness','not-applicable','--summary',desc]
  for p in arts:assert p.exists(),p;args+=['--artifact',str(p)]
  ctl(*args);runs.append(dict(run_dir=run,status='pass',description=desc))
 cp=dict(status='PROGRESS',previous_turn_classification=summary['previous_turn_classification'],current_verified_changes=['All240currentC2graph boundaries and72nativepools reviewed;64exactgraph-root schedules supportconditional endpointaccounting.','Threefileprofile-gated source contract separates pool/ring/norm effects and existing nonalignedfusedbranch.','New F34even-slot state-observability candidate withintegerinvariant/rollbackguard;F18forkdomains/20finitebranches;maininventory34+7.'],summary=str(O/'C2-state-summary-v24.json'),candidate_ledger=str(O/'candidate-profile-ledger-v24.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v24.json'),module_ledger=str(O/'module-profile-ledger-v24.json'),branch_matrix=str(O/'module-phase-branch-matrix-v24.json'),report=str(O/'PROFILE-FINDINGS-v24.md'),source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v24.json'),coverage_gaps=str(O/'module-coverage-gaps-v24.json'),pending_full_scope_domains=remaining,remote_job_status='3CPU-onlyremotejobs and1localabstractreview terminalpass;no targetimplementation/modelimport/NPU/service/healthwork.',model_candidates=34,framework_candidates=7,next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=stamp);write(O/'REVIEW-CHECKPOINT-v24.json',cp)
 for p in[L/'state.json',T/'index.json']:
  d=read(p);assert d['status']in['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=stamp);write(p,d)
 for p in[L/'status.md',T/'status.md']:p.write_text('\n'.join(('- 下一步：'+action)if x.startswith('- 下一步：')else('- 更新时间：`'+stamp+'`')if x.startswith('- 更新时间：')else x for x in p.read_text().splitlines())+'\n')
 (O/'profile-summary-before-v24.json').write_bytes((R/'profile-summary.json').read_bytes());prof=read(R/'profile-summary.json');prof['supplementary_C2_state_v24']=dict(summary=cp['summary'],new_main_candidate='F34',model_candidates=34,framework_candidates=7,current_graph_C2_pairs=240,conditional_endpoint_fraction=0.4375,whole_modules_closed=0,overall_complete=False);write(R/'profile-summary.json',prof)
 for p in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
  d=read(p);d.update(latest_profile_first_review=cp,overall_complete=False,whole_modules_closed=0,updated_utc=stamp);d['next_read_only_action'if p==R/'STATE.json'else'next_action']=action
  if p==W/'STATE.json':d.update(profile_candidate_ledger=cp['candidate_ledger'],profile_module_ledger=cp['module_ledger'])
  if p in[N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[cp['summary'],str(O/'C2-state-opportunity-contract-v24.json'),str(O/'F34-C2-ring-observability-candidate-v24.json'),cp['coverage_gaps']]
  write(p,d)
 compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T));write(O/'task-run-registration-v24.json',dict(runs=runs,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
 arts=[p for p in O.glob('*v24*')if p.is_file()and not p.name.startswith('checkpoint-manifest')]+[N/(k+'.json')for k in labels]+[N/s for s in scripts]+[N/'review_C2_ring_observability_v24.py',N/'zcode-C2-ring-proof-v24-output-20261031.json',N/'finalize_C2_state_opportunities_v24.py',R/'candidates/F34.json',pub,N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'profile-summary.json',L/'state.json',T/'index.json']
 write(O/'checkpoint-manifest-v24.json',dict(generated_utc=stamp,artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in arts],whole_modules_closed=0,overall_complete=False))
 print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v24.json'),runs=len(runs),model_candidates=34,framework_candidates=7,validation=validation,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
