"""Reconcile semantic operator contracts and whole observed universe, preserving uncertainty."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,W,read,write
S=W/'frozen-source-snapshot'

def main():
 assert not (O/'operator-analysis-coverage-ledger-v62.json').exists()
 p=read(O/'state-engram-regime-models-v62.json')
 assert p['regime_count']==276 and p['captured_execution_count']==13704
 for r in p['regimes']:
  if r['module_candidates']==['M14'] and r['operator']=='Slice' and 'embed_gathered' in r['role']:
   r['work_model'].update(source_role='PER_LAYER_HASH_ID_CONTIGUOUS_COPY',fusion='PinnedhostUVAkernelsupportsids_stride_t;hash[T,2,24]slicestride48/inner1canconditionaldirectpass. Otherlookupbranchesretainexplicitcontiguitycontract.',interface_budget_bytes=r['one_read_write_logical_value_bytes'])
 write(O/'state-engram-regime-models-v62.json',p)
 inputs=read(O/'state-engram-operator-inputs-v62.json');old=read(O/'Engram-hash-lookup-publication-and-gate-contract-v30.json')['source_files'];src=[]
 for rel in ['models/deepseek_v41/engram/common.py','models/deepseek_v41/engram/layer.py','models/deepseek_v41/engram/embedding.py','models/deepseek_v41/model.py','models/common/ops/sequence_parallel.py']:
  q=S/'vllm-ascend/vllm_ascend'/rel;sha=hashlib.sha256(q.read_bytes()).hexdigest()
  expected=old.get('/vllm-workspace/vllm-ascend/vllm_ascend/'+rel,old.get(str(q)))
  assert sha==expected,(rel,sha,expected)
  src.append(dict(path=str(q),sha256=sha,existing_pinned_source_reused=True))
 # Operator source contracts only; no new framework/lifecycle survey.
 write(O/'state-engram-operator-contract-v62.json',dict(exact_kernel_files=inputs['files'],existing_pinned_operator_callers=src,
  historical_contract_refs=['C2-state-opportunity-contract-v24.json','C2-ring-observability-integer-review-v24.json','C2-cache-object-and-allocation-contract-v25.json','Engram-hash-lookup-publication-and-gate-contract-v30.json','projection-operator-numerical-contract-v61.json','attention-indexer-native-numerical-contract-v60.json'],
  contracts=[dict(question='DoesC2poolalreadyhavefusednormmathandcanitbedirectlyused?',source='compressor_triton.py:201-321,647-721',answer='NonTOKEN_ALIGNEDbranchalreadyroundsBF16thenFLOATbeforenorm,notunroundedFP32normalization. CurrentTOKEN_ALIGNEDtrueusesdummyoutweight/eps0 andcompletiontokenrows. Reusemathneedsrealweights/eps/independentnormflagandcurrentrowmap;newnumericalpolicyconditional.'),
   dict(question='HowmuchstateIOoccurspercall?',source='compressor_triton.py:85-196,329-364',answer='Oldringonlyfirstresidualpairperoddstartsegment;score3passes/KV1. Updateonlylastmin(n,32)rowsandvalid0<p<N. Allocation2.268GBnottraffic. Preserveint64addressandread-before-write.'),
   dict(question='Canlatentnorm/rope/cachefusewithoutlosingotherconsumer?',source='pinned dsa_v41.py:500-571,compressor.py:52-71 plusv60Indexercontract',answer='C1norm(wkv)orC2norm(pool)latentfirstfeedsIndexerupdate_keys,theninplace64tailRoPEandlongcachewrite. FusionrequirespreRoPEIndexKeyconsumerfork/BF16boundary,slotvalidityandpublication;cannotoverwritelatentearly.'),
   dict(question='Canrequestsearchorcachewriteblindlymerge/reorderwithhash?',source='engram_hash.py:10-167;oldhash_stateforwardv30:95-172',answer='Requestsearchalreadyoncepertoken;sourceexplicitmergedbackendLLVMreleasedbufferabort. Cachewritesbeforefallbackreads;needphysicalslotaliasproof,notlogicalin_batchpreferencealone.'),
   dict(question='Whatishasharithmeticandhistoryreuse?',source='engram_hash.py:86-167',answer='Layerprogram2,shift4each,3ngramlengths×8heads=>8TINT64rollingupdatesand48TINT64remainders,INT32output192T. In-batch>knownwindow>fallback,blockednegative/DEADprefix. Commonhistoryconditional;modcostnotBF16peak.'),
   dict(question='WhichUVAreads/outputzeros/copiescanreduce?',source='engram_lookup.py:68-124;embedding.py:335-369',answer='INT8scalerefs256to8groupbroadcastbutunique32Bsame;MXFP8alreadygrouped. !ownedfallbackrow0canmaskloadswhilefullwritingzeros,benefitunknown. lookupfullwrites3heads×256so176localprezeroselidable. Hostcode/scalesnotHBM. IDstrideparameterallowsperlayerstridedviewintheUVAbranch;othertargetsconditional.'),
   dict(question='WhichexistingEngrammatrixandgateworkisrequired?',source='layer.py:34-57,common.py:47-72',answer='WKV6144→25600alreadycombines4keys+sharedvalueBF16. Rotationblock32FLOATonhiddenforgate;originalhiddenusedresidual. q*kprep3tasks/bodyandrotationcastcanversioncache. FullHtwoRMS/dotbarrier,signbit/NaN/mask/BF16roundpreserved.'),
   dict(question='WhencanTPbank/clone/finalcopyvanish?',source='model.py:1203-1239,1570-1590;sequence_parallel.py:32-43',answer='TPfullheadbankandpreparecopy+SPclonecanconditionaldirectrowlayout,retainingglobalbankallreaders/padding/readiness/clonelifetime. NativefinalhiddenTensorMovecanbefinalgateoutputonlywithaliasmutationproof;graph38TASKbodycontainsnosuchcopyandcannotclaimitsgain.')],
  current_loaded_native_binary_identity_proved=False,broader_framework_audit=False,no_new_Torch_NPU_model_service_capture_implementation=True,overall_complete=False))
 # Reconciliation carries all existing evidence and never closes a module.
 previous=read(O/'operator-analysis-coverage-ledger-v61.json');coverage=deepcopy(previous);rows={r['regime_id']:r for r in coverage['regimes']};new=0;nexec=0
 for r in p['regimes']:
  z=rows[r['regime_id']];assert z['executions']==r['executions'] and z['module_candidates']==r['module_candidates']
  if not z['state'].startswith('SCOPED_'):new+=1;nexec+=z['executions']
  z['state']='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED'
  z['scoped_conditional_review_evidence']+=['state-engram-regime-models-v62.json','state-engram-operator-contract-v62.json','state-engram-fusion-variants-v62.json']
 assert new==263 and nexec==12896
 reviewed=[r for r in rows.values() if r['state'].startswith('SCOPED_')];geom=[r for r in rows.values() if r['state'].startswith('GEOMETRY_')];pending=[r for r in rows.values() if r not in reviewed and r not in geom]
 assert len(reviewed)==448 and sum(r['executions'] for r in reviewed)==89293
 assert len(geom)==5 and sum(r['executions'] for r in geom)==264
 assert len(pending)==1629 and sum(r['executions'] for r in pending)==197393
 coverage.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v61.json',scoped_conditional_review_regimes=len(reviewed),scoped_conditional_review_executions=89293,remaining_regime_reconciliation_count=len(pending),remaining_execution_reconciliation_count=197393,remaining_regimes_by_operator=dict(Counter(r['operator'] for r in pending)),remaining_executions_by_operator=dict(Counter({op:sum(r['executions'] for r in pending if r['operator']==op) for op in set(r['operator'] for r in pending)})),newly_reconciled_regimes=263,newly_reconciled_executions=12896,unchanged_global_uncertainty=dict(ambiguous_module_assignment_executions=3136,missing_PMU=849,phase_unbound_graphs=16),whole_modules_closed=0,overall_complete=False)
 write(O/'operator-analysis-coverage-ledger-v62.json',coverage)
 write(O/'operator-regime-refinement-v62.json',dict(refined_regimes=[dict(regime_id=r['regime_id'],executions=r['executions'],consumer_fusion_review_complete_in_this_turn=True,scope='Explicit operator semanticwork/IO/fusionconditions; notphysicalnumericceiling/allmodulebranches.') for r in p['regimes']],observed_regimes=276,executions=13704,previously_reviewed_projection_regimes=13,previously_reviewed_projection_executions=808,new_review_regimes=263,new_review_executions=12896,whole_modules_closed=0,overall_complete=False))
 ev=['state-engram-regime-models-v62.json','state-engram-graph-operator-chains-v62.json','C2-state-and-pool-costs-v62.json','Engram-hash-UVA-and-gate-costs-v62.json','state-engram-operator-contract-v62.json','state-engram-fusion-variants-v62.json']
 def row(m,i,s,scope=None):return dict(id=f'{m}-OP-{i:02d}',branch=s,state='SCOPED_OPERATOR_CONDITIONAL_FUSION_WORK_IO_REVIEWED;WHOLE_MODULE_OPEN',evidence=ev,gap='Actualtensor/metadata/ownership/parametergeneration/cachetraffic andtypedphysicalcapacity/newkernelprecisionretainconditions;allunobservedapplicablebranchesnotclosed.',scope=scope or 'Fixedcaseoperatorsemanticfamilies;notAPIcatalog/frameworkresearch.')
 m8=[
  'C1BF16linear5120→512,full512RMSthenIndexerpreRoPEforkand64tailRoPElongcache.',
  'C2oneexistingBF16→FLOATcast,twoFLOAT512outputs;v61inputtilejointmodelreused,noadditionalsharecastclaim.',
  'C2ratio2nonresidualstableperchannel2waysoftmax/weightedpool;Ecompletiontokenrows.',
  'Oddstartresidualfirstpairreadsring;B≤requests,score3passes;singleloadratio2conditional16HB/2HBexpbudget.',
  'TOKEN_ALIGNEDsparsecompletionwrites+fullBF16prezero;externalRMSprocessesunwrittenrows,notblinddead.',
  'Pool→BF16round→RMSnativeboundary;oldfusedmathalreadyroundsbutwrongflag/weight/eps/compactrowsfordropin.',
  'Ringupdatevalidpagesandmin(n,32)tail,64bitpageoffset;oldhistoryreadbeforephysicalwrite.',
  'RATIO2evenstateobservability/rollback,allocatorSWAanchornohalfcapacitypromise;externalreadersconditional.',
  'Empty/max_query_len0/invalidpage/used_len0:helperzero/returnandcallernormsemanticsretained.',
  'Poolnormandringmaintenanceforkoronekernel:independentdataedgesversusresource/physicalpageordering.',
  'IndexerKeypreRoPElatentbranchandnorm/RoPE/selectedscatterfusion;selectedupdaterowsnotwholecachedescriptorIO.',
  'C1SWAsharedshape/crosscardproject-before-gatherandcrossiterationringgeneration:ambiguityretained,globalrowsbeforeC2pool.']
 m14=[
  'Current/lookbackdeadmaskEqual/ORandkeepNOTdifferentinputs;validcount/negativeposition/image/dummyguardsnotblanketdedup.',
  'Requestindexbinarysearchoncepertoken;currentbackendexplicitmergereleasedbufferabort,notreadydropinfusion.',
  'BothEngramlayershistorypriorityinbatch>knownwindow>fallback,commonresolutionconditionalparallelism/registertradeoff.',
  'INT64wraprolling8Tupdatesand48Tremainders;exactfixedprimeintegerstrengthreduction,INT32output192T.',
  'SlotcachemappedrawID/DEADwritesbeforehashfallback;physicalslotalias/rollbackandfutureobserverguards.',
  'UVA3formatsBF16/INT8group32/MXFP8group32;B_UVAseparatefromHBM,activeconstexprnotattested.',
  'INT8scales256logicalrefs→8groups,MxFP8already8;uniquehostpayloadsame. !ownedloadskipmuststillfullwritezeros.',
  'VersionedownedIDdedup/devicecache288A→288UforINT8;actualduplicate/hit/capacityunknown,noidentityfromshape.',
  '176perlayerIDSlicecontiguouscopies:UVAids_stride_tpermits[T,2,24]stride48view;otherbranchesconditional.',
  '176lookup-localprezerosfullwriteelision;dummyglobalbank/masktailzerosremainsemanticwrites.',
  'DPgatheralreadyfeedsbothlayers;TPfullheadgather/transpose→head-rowexchangealternative,8xcontentnotnetworktime.',
  'Globalfixedbankcopy,SPpad/chunk/cloneandnativegatefinalcopy:alias/reader/generationpublicationconditions.',
  'WKV6144→25600BF16already4keys+sharedvalue,208densecalls;weight314.5728MB,referencework/IOperM.',
  'q*k2casts+Mul624TASKsandrotationBF16cast208TASKs:versionedpreparedFP32params,notactivationreuse.',
  'Rotatedgateblock32FLOAT,fullHtwoRMSweightedDot/specials;idealvs2passvalueIOandBF16WKVround.',
  'Independenthiddenrotation/normcanstartbeforeWKV;Cube/Vectorcompetition,originalhiddenresidualandV4wayfanout.',
  'WKVpanels→gatepartialsanddirectoutputstore:fullHbarrier/partialworkspace,round/reduction/signbit/NaNmaskguards.',
  'HeadshardedpartialWKVkeepspercardFLOPs,weights/8addsFP32reduceplane200/3vsbesthead-rowinputcontent.',
  'Unrotatednativefusedgate,TP1/CPU/deviceINT8,slotless/dummy/empty/validcount/DPsharedbranchesremainseparatefromcapturedrotated/UVAcase.']
 write(O/'M08-M14-operator-branch-inventory-v62.json',dict(modules={'M08':[row('M08',i+1,s) for i,s in enumerate(m8)],'M14':[row('M14',i+1,s) for i,s in enumerate(m14)]},semantic_families=len(m8)+len(m14),scope='Conditionaloperatoranalysisnotwholemoduleclosure. Priorbranchinventoriesarchivedonpersistence;M01/M21/M22/M25andpriorM07/M09/M10/M11semanticbranchesprotected.',overall_complete=False))
 write(O/'state-engram-local-validation-history-v62.json',dict(resolved_local_attempts=[dict(attempt=1,status='error',reason='Explicitcustomkernelmodeldispatchomitted_write_hash_cache_kernel;conservationandgraphassertionshadpassed;noartifactwritten.'),dict(attempt=2,status='error',reason='Dispatchwhitelistthenexposed_hash_ids_kernel;addedexactname,noartifactwritten.'),dict(attempt=3,status='error',reason='ConsolidationimportedSfromhelperthatonlyexportsW;resolvedS=W/frozen-source-snapshotbeforeexecution.')],resolved_by='Finalanalysis13704exactnativeTASKkeys/276regimesand240/160graphchainsPASSED;allcustomkernelsnowexplicitparameterizedmodels. Consolidation263newregimes/12896execand31semanticfamiliesPASSED.',target_or_server_failure=False,overall_complete=False))
 print(dict(new_regimes=new,new_executions=nexec,reviewed_regimes=448,reviewed_executions=89293,pending_regimes=1629,pending_executions=197393,semantic_branches=len(m8)+len(m14)))

if __name__=='__main__':main()
