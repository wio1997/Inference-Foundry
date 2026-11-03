"""Maintain the full observed-regime ledger with explicit review versus geometry-only states."""
from pathlib import Path
from collections import Counter,defaultdict
import csv
from finalize_origin_coverage_v9 import N,O,read,write

def main():
 assert not (O/'operator-analysis-coverage-ledger-v61.json').exists()
 p=read(O/'projection-operator-numerical-contract-v61.json')
 p['facts'].append('QR DynamicQuant beforeQ_B/IndexQ usesoneFLOATrowScale over1280;customIndexerheadquantafterIndexQuses32heads×128andFP16roundedscales. Theyaredifferentoperators andmustnotbeinterchanged.')
 write(O/'projection-operator-numerical-contract-v61.json',p)
 c=read(O/'projection-fusion-and-partition-costs-v61.json')
 c['row_domain_countermodel']['local_joint_QA_KV_before_redistribution_ops']=2*3*5120*(1280+512)
 c['row_domain_countermodel']['separate_to_local_joint_percard_work_ratio_under_replicated_weight_domain']=3
 write(O/'projection-fusion-and-partition-costs-v61.json',c)
 e=['projection-work-and-reference-bounds-v61.json','projection-graph-chains-v61.json','projection-fusion-and-partition-costs-v61.json','projection-operator-numerical-contract-v61.json','projection-fusion-variants-and-interactions-v61.json']
 common='Actualphysicalcache/typedcapacity/dequant/Vector/internaltilecost andnewkernelallconsumer/precision/readiness requirements remainexplicit;conditionalanalysisdoesnotrequireimplementation.'
 def row(mid,i,branch,gap=common,evidence=e):return dict(id=f'{mid}-OP-{i:02d}',branch=branch,state='SCOPED_OPERATOR_FORMULA_FUSION_AND_COST_REVIEWED;WHOLE_MODULE_OPEN',evidence=evidence,gap=gap,scope='Fixed910C/CANN9.2case,source-conditionedoperatorfamilies;notallsourceguard/nativebinaryattestation.')
 m7=[
  row('M07',1,'Q_A BF16linear5120→1280;4367executions,localM2/3/800/816/830/834densework/weightIOmodels.'),
  row('M07',2,'Q_A→BF16RMS1280→DynamicQuantFLOATrowscale→INT8Q_B;RMSrstd/roundedQRrowamax dependencies constrainpanelstreaming.'),
  row('M07',3,'INT8Q_B1280→32768,BF16out;FRACTAL_NZnumel/scalesperrow andperoutput kept,64tailRoPEepilogueconditional.'),
  row('M07',4,'SWA KV5120→512BF16globalrowdomain;C1source20twoidenticalshapes source-order derived,original160M07|M08assignments retained.'),
  row('M07',5,'KV BF16RMS512→64tailRoPE→SWA cacheScatter;actualslots/masks/globalrows andfuturecachewrite remainlive.'),
  row('M07',6,'Q_B andIndexQ consume sameqrsource;640exactgraphduplicateDQpairs,existingquantmethodPreparedLinearInput boundary supportsconditionalproducerreuse.'),
  row('M07',7,'Indexweightprojectiondependsquery_hidden independentlyofQ_B/IndexQ;earlyorjointQA+32weightcolumns keepsseparatenorm/outputdomains.'),
  row('M07',8,'UnequalQ_A3/KV24anddirectlocal/global/paddedrowcounts excludeblindfullglobalconcat;projectlocal→communicateroleoutputs isseparateconditionalrepartition.'),
  row('M07',9,'Emptylocalquery canstillrequireKV/cache andpostcollectiveoutput;57output-minusQAcount isnotdeadwork.',evidence=e+['cache-only-KV-branch-reconciliation-v18.json']),
  row('M07',10,'Crosslayer/iteration weightshape/pointerreuse isnotquery/weightvalueidentity;existingone-timeNZpacking andcurrentCube/Vectoroverlap arebaseline.',gap=common+' OtherunobservedapplicableQ/KV/norm/Rope branchesandallVectorregimeceilings notclosed.'),
 ]
 m11=[
  row('M11',1,'All7WO_A TransposeBatchMatmulregimes/4424calls G1:permutes singletonaxes,logicalM=T,K4096,R1024;ordinaryGEMMspecialization conditional.'),
  row('M11',2,'WO_B BF16linear1024→5120;smallrowsuseMatMulV2,large6400/6528/6672MatMulV3. Sameoperatorfamilycostbothdispatchesincluded.'),
  row('M11',3,'ExistingTransposeBatchMatmulalreadyfusesAPIpermutations;weightstride/orientation andviewoutputlifetime retained;noextraTransposekernelinvented.'),
  row('M11',4,'WO_A→WO_B rank1024panels preserveBF16intermediate;logical4MRread/writecanbeconditionalreduced,butsharedCube capacitypreventsfree2GEMMoverlap.'),
  row('M11',5,'Staticweightprecomposition4096×5120 failsBF16roundwitness andraisescompute/weights20/9;IOthreshold5632 alone isnotperformancecase.'),
  row('M11',6,'WO_B→ReduceScatter→localHCPost:perrankBF16partialmustremainrounded beforecollectivereduction;tilepanelorder/ownership/collectiveprotocol allrequired.'),
  row('M11',7,'Actualglobal6668versuspadded6672pluslocal830/834rows keptdistinct. Localzeroquerydoesnotpermitremovingallcard/globaloutputprojections.'),
  row('M11',8,'G>1/full-gatherconditionalpathkeptseparatefromcapturedG1;A5FP8/packedpathnotappliedtofixed910C/BF16capture.',gap=common+' Unobservedapplicablefull-gather/outputbranches notprovenbyallG1samples.'),
  row('M11',9,'InverseRoPE64tail/send-layout/AlltoAll boundary reusesv60contracts;rankheadgroups,alias/padrows andfinalprojectedoutputpublication persist.',evidence=e+['attention-indexer-native-numerical-contract-v60.json']),
 ]
 write(O/'M07-M11-operator-branch-inventory-v61.json',dict(modules={'M07':m7,'M11':m11},scope='19semanticoperatorfamilies;historypreserved,wholemoduleopen. M09/M10v60semanticbranches retained;onlyreview_historyextendedwhereupdated.',overall_complete=False))
 universe=read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes'];lookup={r['regime_id']:r for r in universe};assert len(lookup)==2082
 reviewed={};geometry={}
 def add(rid,ev,executions,complete=True):
  assert rid in lookup and lookup[rid]['executions']==executions
  (reviewed if complete else geometry).setdefault(rid,[]).append(ev)
 for r in read(O/'shared-expert-shape-ceiling-v57.json')['all_captured_M13_regimes']:add(r['regime_id'],'shared-expert-shape-ceiling-v57.json',lookup[r['regime_id']]['executions'])
 for fn in ['operator-regime-refinement-v58.json','operator-regime-refinement-v60.json','operator-regime-refinement-v61.json']:
  for r in read(O/fn)['refined_regimes']:add(r['regime_id'],fn,r['executions'],r.get('consumer_fusion_review_complete_in_this_turn',True))
 mc=read(O/'MC2-operator-profile-v56.json');sh={r['input_shapes'] for r in mc['shape_timings']}
 for r in universe:
  if r['operator']=='DispatchFFNCombine':
   assert r['signature']['inputShapes_text'] in sh;add(r['regime_id'],'MC2-operator-profile-v56.json',r['executions'])
 assert len(reviewed)==185 and len(geometry)==5 and not set(reviewed).intersection(geometry)
 with (O/'operator-dense-reference-roofline-v55.csv').open() as f:dense={r['regime_id'] for r in csv.DictReader(f)}
 assert len(dense)==87
 rows=[];pending=Counter();pexec=Counter();reviewExec=geometryExec=0
 for r in universe:
  rid=r['regime_id'];full=rid in reviewed;geo=rid in geometry
  state='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED' if full else 'GEOMETRY_ONLY_OTHER_ROLE_CONSUMER_FUSION_PENDING' if geo else 'REGIME_RECONCILIATION_WITH_EXISTING_FAMILY_EVIDENCE_PENDING'
  if full:reviewExec+=r['executions']
  elif geo:geometryExec+=r['executions']
  else:pending[r['operator']]+=1;pexec[r['operator']]+=r['executions']
  rows.append(dict(regime_id=rid,operator=r['operator'],module_candidates=r['module_candidates'],role=r['role'],origin=r['origin'],context=r['context'],executions=r['executions'],
   state=state,scoped_conditional_review_evidence=reviewed.get(rid,[]),geometry_only_evidence=geometry.get(rid,[]),generic_dense_reference_present=rid in dense,
   full_numeric_actual_hardware_ceiling_proved=False,all_model_branches_or_whole_module_closed=False))
 assert reviewExec==76397 and geometryExec==264
 assert sum(r['executions'] for r in rows)==286950 and len(rows)==2082
 write(O/'operator-analysis-coverage-ledger-v61.json',dict(parent_universe='operator-shape-ceiling-ledger-v55.json',observed_regimes=2082,observed_operators=83,observed_executions=286950,
  regimes=rows,scoped_conditional_review_regimes=185,scoped_conditional_review_executions=reviewExec,geometry_only_other_role_regimes=5,geometry_only_other_role_executions=geometryExec,
  remaining_regime_reconciliation_count=1892,remaining_execution_reconciliation_count=286950-reviewExec-geometryExec,
  remaining_regimes_by_operator=dict(pending),remaining_executions_by_operator=dict(pexec),generic_dense_reference_regimes=87,
  interpretation='Full2082observedregimesindexedwithoutpromotinggenericdenseformula/sourcefamily/historicalcandidateevidencetofullyanalyzedfusion. Pendingmeanscurrentexplicitregime-indexneedsreconcileexistingfamilyevidence,notclaimallpriorcandidateanalysisabsent. Nativeimplementationabsencealoneisnotunfinishedconditionalanalysis. Thisobserveduniverseisnotallmodelbranches/fullscopeproof.',
  physical_capacity_or_all_numeric_ceilings_complete=False,whole_modules_closed=0,overall_complete=False))
 print('2082regimes/286950executions conserved:185scopedconditionalreview,5geometry-only,1892pendingexplicitregime/familyreconciliation;19M07/M11families.')

if __name__=='__main__':main()
