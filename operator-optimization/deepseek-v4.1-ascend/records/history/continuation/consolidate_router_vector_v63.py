"""Promote only answered conditional operator regimes, preserving MemSet and whole-model gaps."""
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
import json,sqlite3,hashlib
from finalize_origin_coverage_v9 import N,O,W,read,write
from mirror_saved_profile_core_v53 import CACHE
S=W/'frozen-source-snapshot'

def main():
 assert not (O/'operator-analysis-coverage-ledger-v63.json').exists()
 p=read(O/'router-vector-regime-models-v63.json');assert len(p['regimes'])==145
 # Exhaust the retained attribute/launch records for the unpriced operations.
 m=read(O/'router-MemSet-profile-insufficiency-v63.json');grouped=defaultdict(list)
 for r in m['native_records']:grouped[r['database_index']].append(r)
 attrs=Counter();names=Counter()
 for i,records in grouped.items():
  e=read(CACHE/f'analysis/native-execution-identity-v14/db-{i:02d}.json');remote=Path(e['database_path']);j=max(k for k,x in enumerate(remote.parts) if x=='analysis');path=CACHE/Path(*remote.parts[j:]);db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True);db.execute('pragma query_only=1');strings=dict(db.execute('select id,value from STRING_IDS'))
  for r in records:
   v=strings.get(r['definition']['attrInfo']);attrs[str(v)]+=1;r['resolved_attrInfo']=v
   ns=[strings.get(z[0]) for z in db.execute('select name from CANN_API where connectionId=?',(r['TASK']['connectionId'],))];r['linked_CANN_API_names']=ns
   names.update(ns)
  db.close()
 assert attrs=={'':960} and names=={'launch':960}
 m.update(resolved_attribute_value_counts=dict(attrs),linked_CANN_name_counts=dict(names),operand_gap_confirmed_after_raw_attribute_and_launch_lookup=True)
 write(O/'router-MemSet-profile-insufficiency-v63.json',m)
 contract=read(O/'router-vector-numerical-contract-v63.json');pins=[]
 old=read(O/'F04-precision-routing-and-row-contract-v29.json')['source_files']
 for rel in ['vllm_ascend/ops/fused_moe/router/fused_topk_router.py','vllm_ascend/ops/fused_moe/prepare_finalize.py','vllm_ascend/ops/fused_moe/fused_moe.py','vllm_ascend/device/device_op.py','vllm_ascend/models/deepseek_v41/model.py']:
  path=S/'vllm-ascend'/rel;sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==old['/vllm-workspace/vllm-ascend/'+rel],rel;pins.append(dict(path=str(path),sha256=sha,previous_166_hash_binding='F04-precision-routing-and-row-contract-v29.json'))
 contract.update(exact_source_inputs='router-vector-operator-inputs-v63.json',prior_pinned_operator_callers=pins,MemSet_unknown_retained=True)
 write(O/'router-vector-numerical-contract-v63.json',contract)
 c=deepcopy(read(O/'operator-analysis-coverage-ledger-v62.json'));rows={r['regime_id']:r for r in c['regimes']};new=[];unresolved=[]
 evidence=['router-vector-regime-models-v63.json','router-vector-graph-chains-v63.json','router-vector-numerical-contract-v63.json','router-vector-fusion-costs-v63.json','router-vector-fusion-variants-v63.json']
 for r in p['regimes']:
  row=rows[r['regime_id']];assert row['state'].startswith('REGIME_') and row['executions']==r['executions']
  if r['consumer_fusion_review_complete_in_this_turn']:
   row.update(state='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED');row['scoped_conditional_review_evidence']+=evidence;new.append(r)
  else:
   assert r['operator']=='MemSet';row.update(pending_reconciliation_reason='PROFILE_NO_OPERAND_ADDRESS_LENGTH_OR_DESTINATION_KIND',partial_review_evidence=['router-MemSet-profile-insufficiency-v63.json']+evidence);unresolved.append(r)
 assert len(new)==142 and sum(r['executions'] for r in new)==74842 and len(unresolved)==3
 reviewed=[r for r in rows.values() if r['state'].startswith('SCOPED_')];geometry=[r for r in rows.values() if r['state'].startswith('GEOMETRY_')];pending=[r for r in rows.values() if r['state'].startswith('REGIME_')]
 assert len(reviewed)==590 and sum(r['executions'] for r in reviewed)==164135
 assert len(geometry)==5 and sum(r['executions'] for r in geometry)==264
 assert len(pending)==1487 and sum(r['executions'] for r in pending)==122551
 c.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v62.json',scoped_conditional_review_regimes=590,scoped_conditional_review_executions=164135,remaining_regime_reconciliation_count=1487,remaining_execution_reconciliation_count=122551,remaining_regimes_by_operator=dict(Counter(r['operator'] for r in pending)),remaining_executions_by_operator={op:sum(r['executions'] for r in pending if r['operator']==op) for op in sorted(set(r['operator'] for r in pending))},newly_reconciled_regimes=142,newly_reconciled_executions=74842,explicit_profile_insufficient_regimes=3,explicit_profile_insufficient_executions=960)
 write(O/'operator-analysis-coverage-ledger-v63.json',c)
 write(O/'operator-regime-refinement-v63.json',dict(refined_regimes=[dict(regime_id=r['regime_id'],executions=r['executions'],consumer_fusion_review_complete_in_this_turn=r['consumer_fusion_review_complete_in_this_turn'],scope='Operatorwork/logicalIO/fusionconditionsreviewed;MemSetdestinationandlengthunresolvedandnotpromoted.') for r in p['regimes']],modeled_or_audited_regimes=145,executions=75802,new_review_regimes=142,new_review_executions=74842,MemSet_pending_regimes=3,MemSet_pending_executions=960,whole_modules_closed=0,overall_complete=False))
 matrix={r['id']:r for r in read(O/'module-phase-branch-matrix-v62.json')['modules']}
 def row(mid,i,s,ev=evidence):return dict(id=f'{mid}-OP-{i:02d}',branch=s,state='SCOPED_OPERATOR_CONDITIONAL_WORK_IO_FUSION_REVIEWED;WHOLE_MODULE_OPEN',evidence=ev,gap='Actualvalue/metadata/alias/generation,typedphysicalcache/FLOAT/Vector/sort/specialcapacity andnewkernelprecision/readiness/collectivecontractretainconditions;unobservedapplicablebranchesnotclosed.',scope='Fixedcaseoperatorsemantics;noAPIcatalog/frameworkresearch.')
 m7=deepcopy(matrix['M07']['branches']);assert len(m7)==10
 for x in m7:
  if x['id'] in ['M07-OP-02','M07-OP-05','M07-OP-06']:
   x['evidence']+=evidence;x['review_v63']='Explicit145regimepopulationcoversRMS/DQ/tailRoPE/selectedscattercostandRMSBF16roundingwitness;QRnonadditivejointbudgetandactualsourceparitysign/CAST_RINT.'
 m7.append(row('M07',11,'AttentioninputRMS5120/Q_RMS1280/KV_RMS512specificMdomainswork/IO;livefullrowBF16boundariesandauxrstd,normproducerfanoutandGEMM/quantconsumerinterfacebudgets.'))
 m11=deepcopy(matrix['M11']['branches']);assert len(m11)==9
 m11[-1]['evidence']+=evidence;m11[-1]['review_v63']='Exactquery/outputinverseRoPEtail-onlysourcebroadcast/signflip/RINTand3200sendlayoutrecords;noextraNegorfull512RoPEtrafficinvented.'
 m11 += [row('M11',10,'960targetWO_BresultTensorMoves:directprojectedbankwriteconditional4THcopybudget,alloutputaliases/lifetimes/RSBF16partialroundretained.'),row('M11',11,'312RSsuffixZero+ConcatpairsM10→P16/M20→P24:directPbankprefixandsuffixzeros,4PHConcatbudgetandunchangedcollectivecounts.'),row('M11',12,'118CPfullzerosand40activeprefix830copies;sourceconditionedemitactive+zerosuffix/sendlayout,emptylocal/globalquerycasesandnonadditiveinterfacesretained.')]
 descriptions=[
  'RmsNormCast4472callsalreadyfuseddualBF16/FLOAT;FP32wideningofroundedBF16,fullliveBF16expert/sharedpath.',
  'RouterFP32dense5120→384target/128draft,M2/3/800/816/834domains,FLOATactiveprecision/capacityunbound.',
  'normType2FP32Exp→+1→Ln→Sqrt;vision/textbiasonlyforrank,unbiasedgatheredweightsselectedsum+eps/Div/scalingevenrenorm0.',
  'Currenttypedtid2eidundefined,G1sourceconsistentwithout-groupsort;hashmappingandgeneralgroup/fallbackbranchesdistinct.',
  'AlreadyfusedfullUBsortE=384/128,k6/3;fullrowepilogue/columnlocalTopK12Bvs8BmergecostandnativeTie/NaNgates.',
  'FP32normplanewideninsideGEMM,fullrowRMSorKsplitpartialworkpriced;rowRMS/nonlinear/BF16cancellationexcluded.',
  'TokenIDINT32→INT64,-1-onlyzeroCast/Equal/Fill/Select;othernegativevisionIDsandrawinputsretainsemantics.',
  'CrosslayerIDgenerationCSEorTopKinline,zero-pad/rowselectcommutation;allrowplans/parameterbanks/collectivesandinvalidationpreserved.',
  'MC2hidden/logits800→816paddingandBOOL→INT8pad→BOOL/IDzeroalign;directpaddedproducer/packingconditional.',
  'ExplicitfalsepaddingTopKworkcanconditionalpruneonlywithsafeoutputIDs/weightsandallmaskedconsumers;M816nativepopulationnotallM800pads.',
  '3MemSetregimes960executions:allattrInfoemptyandCANNlaunchonly,N/Adest/length;explicitpendingnobyte/traffic/deadwriteguess.',
  'outflagfalsefullscoreoutputnotGMwritteninsource;outOutQueueE4UBstillallocated,conditionalcapacityspecializationnotnewHBMwin.',
  'RoutedMC2DispatchFFNCombinealreadyfusesdispatchquant/GMM1/activation/GMM2/combine;v56receiverrows/workspace/internalstageevidence retained.',
  'Shared/routedbranchreadiness/scaling/HCoutputandfiniteCube/Vector/BWcompetition;nofreeoverlaporcrosslayerhiddenreuse.',
  'Currenttarget/draft/dummy/empty/paddingcasesandotherfallback/allgather/hash/vision/group/quantcommunicationfamiliesremainseparate,allmodelbranchesnotclosed.']
 m12=[row('M12',i+1,s) for i,s in enumerate(descriptions)]
 m12[10]['state']='PROFILE_INSUFFICIENT_DESTINATION_LENGTH;EXPLICIT_PENDING';m12[10]['evidence']=['router-MemSet-profile-insufficiency-v63.json']+evidence
 m12[12]['evidence']+=['MC2-operator-profile-v56.json']
 write(O/'M07-M11-M12-operator-branch-inventory-v63.json',dict(modules={'M07':m7,'M11':m11,'M12':m12},M07_families=11,M11_families=12,M12_families=15,total_families=38,new_or_extended_semantic_inventory_not_whole_module_closure=True,overall_complete=False))
 write(O/'router-vector-local-validation-history-v63.json',dict(resolved_local_attempts=[dict(status='error',reason='FirstschemaauditderivedlocalDBpathfromfirstanalysisdirectoryinremoteabsolutepath;nestedanalysisprefixcausedread-onlyopenfailure. Noartifactswritten;correctedtousefinalanalysiscomponent.')],resolved_by='32immutableDBschemas/960emptyattrInfo+launchlinks,145exactregimecounts75802TASKs/3200graphchainsand142review/3pendingPASSED.',target_or_server_failure=False,overall_complete=False))
 print(dict(observed_regimes=2082,reviewed_regimes=590,reviewed_executions=164135,geometry_only=5,pending_regimes=1487,pending_executions=122551,MemSet_pending=3,semantic_families=38))

if __name__=='__main__':main()
