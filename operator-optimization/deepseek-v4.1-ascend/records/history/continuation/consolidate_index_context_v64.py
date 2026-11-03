"""Index/aux-context evidence reconciliation without promoting unknown capacities or full modules."""
from copy import deepcopy
from collections import Counter
from finalize_origin_coverage_v9 import N,O,read,write

def main():
 assert not (O/'operator-analysis-coverage-ledger-v64.json').exists()
 p=read(O/'index-context-regime-models-v64.json');c=deepcopy(read(O/'operator-analysis-coverage-ledger-v63.json'));rows={r['regime_id']:r for r in c['regimes']}
 e=['index-context-regime-models-v64.json','index-context-native-chains-v64.json','index-context-operator-contract-v64.json','index-context-fusion-costs-v64.json','index-context-fusion-variants-v64.json']
 new_pending=[];old_geometry=[]
 for r in p['regimes']:
  x=rows[r['regime_id']];assert x['executions']==r['executions'] and x['module_candidates']==r['module_candidates'] and r['consumer_fusion_review_complete_in_this_turn']
  assert x['state'].startswith(('REGIME_','GEOMETRY_'))
  (old_geometry if x['state'].startswith('GEOMETRY_') else new_pending).append(r)
  x.update(state='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED');x['scoped_conditional_review_evidence']+=e
 assert len(new_pending)==237 and sum(r['executions'] for r in new_pending)==15712
 assert len(old_geometry)==5 and sum(r['executions'] for r in old_geometry)==264
 reviewed=[r for r in rows.values() if r['state'].startswith('SCOPED_')];pending=[r for r in rows.values() if r['state'].startswith('REGIME_')];geometry=[r for r in rows.values() if r['state'].startswith('GEOMETRY_')]
 assert len(reviewed)==832 and sum(r['executions'] for r in reviewed)==180111
 assert len(pending)==1250 and sum(r['executions'] for r in pending)==106839 and not geometry
 c.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v63.json',scoped_conditional_review_regimes=832,scoped_conditional_review_executions=180111,geometry_only_other_role_regimes=0,geometry_only_other_role_executions=0,remaining_regime_reconciliation_count=1250,remaining_execution_reconciliation_count=106839,remaining_regimes_by_operator=dict(Counter(r['operator'] for r in pending)),remaining_executions_by_operator={op:sum(r['executions'] for r in pending if r['operator']==op) for op in sorted(set(r['operator'] for r in pending))},newly_reconciled_regimes=242,newly_reconciled_executions=15976,new_pending_regimes_reconciled=237,new_pending_executions_reconciled=15712,prior_geometry_promoted_to_role_conditional_review=5,prior_geometry_executions_promoted=264)
 assert c['explicit_profile_insufficient_regimes']==3 and c['explicit_profile_insufficient_executions']==960
 write(O/'operator-analysis-coverage-ledger-v64.json',c)
 write(O/'operator-regime-refinement-v64.json',dict(refined_regimes=[dict(regime_id=r['regime_id'],executions=r['executions'],consumer_fusion_review_complete_in_this_turn=True,scope='ExplicitIndex/Draftcontextoperatorwork/logicalIO/fusionconditions;physicalnumericcapacity/allmodulebranchesnotclosed.') for r in p['regimes']],reviewed_regimes=242,executions=15976,previous_pending_regimes=237,previous_pending_executions=15712,previous_geometry_only_regimes=5,previous_geometry_only_executions=264,whole_modules_closed=0,overall_complete=False))
 old={m['id']:m for m in read(O/'module-phase-branch-matrix-v63.json')['modules']};m9=deepcopy(old['M09']['branches']);assert len(m9)==10
 for x in m9:
  if x['id'] in ['M09-OP-04','M09-OP-05','M09-OP-06','M09-OP-08','M09-OP-10']:
   x['evidence']+=e;x['review_v64']='72remainingstrictM09regimes6040execand320IndexKeygraphchains;FLOAT-before-FP16keyscalewitness,2^-6weightconvert,dualcache/coherency/TPkeyownerandfreshcandidateoutABIbudgetexplicit.'
 def row(i,s):return dict(id=f'M18-OP-{i:02d}',branch=s,state='SCOPED_OPERATOR_CONDITIONAL_WORK_IO_FUSION_REVIEWED;WHOLE_MODULE_OPEN',evidence=e,gap='ActualI/validslot/position/group/alias/valueandparametergeneration,allbankobservers/nativeprecision/resource/collectiveprotocolandunobservedapplicableconditionsremainexplicit.',scope='Fixed910C/CANN9.2caseoperatorfamilies,notAPIcatalog/frameworkaudit.')
 bs=[
  'ThreeHCmeansBF16atnativecheckpointsbeforeblocks37/38/39;separatemeanwork/readbudgetandconditionalHcPresharedinputload/multioutputreadiness,notweightedYreuse.',
  'AuxSPgathersNphysicalrowsandafter-samplingorderedIselection;late-selectivegatherkeeps640Wshardsbutdeferscommunication,raggedUvsstatic8max(U_p)andlostoverlapexplicit.',
  'Threeorderedauxsources/Concat→single15360maincontraction;indirectinputandFP32crosscheckpointpartialsconditional,noBF16roundpartials.',
  '88mainBF16M×15360→640perrankprojections,M12/24/6400/6528/6668geometry/work/weightsandpublishedconditionalreferencecost.',
  'FeatureAllGather[8,M,640]→Transpose→full5120RMS;stridednormcanavoid4MHtransposebytes,8independent640normsexcluded.',
  'ActualDSparkpublisherNc-prefixpersistentcontextbankandunchangedtail/allreaders;24positivecopyevidenceandmetadata274remainseparatefunctionalpopulations.',
  'Complete-weightrowownerWmain157.2864MBvs19.6608MBshard;mainN/M+KV/8compute,N24/M12combinedops11/9notalwaysbetter.',
  'Featureglobalnormstats→partialFP32KV→rowRS→BF16full512norm/RoPE→KVGather;mainweightsunchanged,matrixops11/18,8071Mif8divisibleotherwise8064P+7M. P12→16communicationworse;fullbankretentionaddAGback.',
  'ThreeKVlayer-specificweights/norms/caches,312source-order30TASKcycles;1536jointGEMMinputreuse4M5120B,three512norms/BF16barriersandfirstlayerreadinessretained.',
  'PureRoPElookup4Index+4IndexCheck/cycle,2C/Sconfigpairs;union/filter/hoistbyregistry/config/positiongeneration,negativeindexwrap/bounds/errorsemanticsretained.',
  'M18FLOATC/SnativeBF16casts→builtinRotary;BF16C/SroundcannotdroporreplacebyM09FLOATcospath. ExplicitsharedBF16banknewABIseparatefromFP32lookuphoist.',
  'TailSlice/Rotary/ViewCopymutates64tailand448+64Concat/cacheassembly;baseM512ViewCopydescriptorisnotfullcopytraffic,alloriginalviewobserversretained.',
  'Standard1Dslots→int64compare/div→INT32pair/invalid(-1,-1),allnegativeinvalid;slotmath124Minterfacebudgetinline,notwholecachedescriptorbytes.',
  'Per-layer_layer_group_idx/slotbank/B/epochCSE;threecachewritesremain,actualgroupsunknown,duplicate-slotsequence/readinessrequired.',
  'None/emptyhidden/positions/global-slotsbaselineguards;newperlayerNone/emptyorinvalidslotprivateprojectionpruneonlyunderpurevalidnormaldomain/allreaders.',
  'CrossiterationdeltaKVreusekeysrequest/acceptedtargetaux/positions/weights/normRope/group/page/slotgeneration/lastreader;existingKVcachebaseline,nosamepointer/slotreuseproof.',
  'Contextfeatureweightviewsworking-setonly;attn.wkvfullparametersstillqueryKVconsumers,noVRAM/8promise. Querypreparation/metadataandTP-DCProwmapsretained.',
  'K0proposeearlyreturnandotherdraft/dummy/SP/nonSP/packedguards:skippingtargetauxneedspriorK0andallobservers/collectivegroupinactive;currentK5capturedoesnotattestK0.']
 m18=[row(i+1,s) for i,s in enumerate(bs)]
 write(O/'M09-M18-operator-branch-inventory-v64.json',dict(modules={'M09':m9,'M18':m18},M09_retained_and_extended_families=10,M18_semantic_families=18,total_families=28,scope='v60M09familiesretained/extended;oldM18chronologicalrecordsarchivedbyfinalizer. M06HCsemanticbranchesnotreplaced,onlysharedinputreviewhistoryextended.',whole_modules_closed=0,overall_complete=False))
 write(O/'index-context-local-validation-history-v64.json',dict(resolved_local_attempts=[dict(status='error',reason='Firstauxgraphwindowassumedcheckpointbefore36/37/38fromconfignumbering. ExactnativeTASKtimestamps/markerboundsshow37/38/39;correctedbeforeanyartifactwritten.'),dict(status='error',reason='Modelsourcepinlookupusedsource_filesforv38whosefieldissource_pins. Onlyuncheckpointedcostfilehadbeenwritten;correctedfieldandregeneratedmodels.')],resolved_by='242regimes15976uniqueTASKs/320Keychains/240auxchains/31230TASKKVcycles/88mainpipelinesandallcost/padding/conservationassertionsPASS.',target_or_server_failure=False,overall_complete=False))
 print(dict(regimes=242,executions=15976,reviewed_total=832,reviewed_exec_total=180111,pending_total=1250,pending_exec_total=106839,geometry_only_remaining=0,semantic_families=28))

if __name__=='__main__':main()
