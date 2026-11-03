"""Add operator semantic coverage and explicit cross-iteration/cross-card limits."""
from pathlib import Path
import hashlib,re
from finalize_origin_coverage_v9 import N,O,read,write

def main():
 assert not (O/'M09-M10-operator-branch-inventory-v60.json').exists()
 serve=N/'original-profile-cache-v53/runtime/scripts/serve.sh';b=serve.read_bytes()
 assert hashlib.sha256(b).hexdigest()=='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
 assert re.search(r'--max-model-len\s+34816',b.decode())
 models=read(O/'attention-indexer-work-traffic-models-v60.json')
 for m in models['models']:
  for r in m['indexer_examples']:
   r['source_weight_expansion_GM_interface_bytes_per_one_pack_per_query']=r.pop('source_weight_expansion_GM_interface_bytes')
   r['weight_pack_scope']='ProcessVec0 runsatfirstS2innerloop perparticipatingcorepartition;QKpreloadpartitionsmayrepeatquerypacks. This isonelogicalpack/query,not totalactualtraffic.'
 write(O/'attention-indexer-work-traffic-models-v60.json',models)
 c=read(O/'attention-indexer-native-numerical-contract-v60.json')
 c['model_config']['max_model_len_case_binding']=dict(path=str(serve),sha256=hashlib.sha256(b).hexdigest(),argument=34816)
 c['witnesses']['cross_iteration_query_change']=dict(keys=[[1,0],[0,1]],old_query=[1,0],new_query=[0,1],old_scores=[1,0],new_scores=[0,1],
  implication='Evenwithunchangedoldkeys,changingquery reversesTopK. Pointer/shape/cacheappendalone cannotjustifyreuseoldQLIscores orupdatingonlynewkeys. ReLU/positiveweightonehead example;not native execution.')
 c['crosscard_work_partition_counterexample']=dict(current='Mlocal queryrows x fullLkeys',repartition='(TP*Mlocal)globalqueries x (L/TP)keyspercard',
  percard_densework_ratio=1,qualification='Conditional algebra foralreadyquery-shardedcase;notassertionofallranksactualtensoridentity. Full32heads perrankalone isnot proofduplicatequerywork.')
 write(O/'attention-indexer-native-numerical-contract-v60.json',c)
 v=read(O/'attention-indexer-fusion-reorganization-variants-v60.json')
 v['variants'].append(dict(id='F12-CROSSITER-DELTA',proposal='ReuseoldperkeyQLIscores andscoreonlyappendedkeys.',decision='REJECT_UNCONDITIONAL_CONDITIONAL_IDENTITY_DOMAIN',
  conditions='Oldquantizedquery/roundedqscale/FP16weights+product,oldkey/scalegenerations,visiblemask/candidate/pin/outputoffset/tiepolicy mustallmatch;appendedkeys/rollbackthenupdateexacttopK. Newtokennormallychangesquery,so cachegrowthalone doesnotqualify.',
  work_delta='Explicit2keyquery-change witness reversesTopK. Noautomaticcrossiterationfullscanremoval orscore reusegainclaimed.'))
 write(O/'attention-indexer-fusion-reorganization-variants-v60.json',v)
 e=['attention-indexer-profile-and-edges-v60.json','attention-indexer-work-traffic-models-v60.json','attention-indexer-native-numerical-contract-v60.json','attention-indexer-fusion-reorganization-variants-v60.json']
 def row(mid,i,branch,gap,evidence=e,scope='Fixed910C/CANN9.2case;capturednativeprofile+source-conditionedoperator analysis. Fullmoduleclosureunproved.'):
  return dict(id=f'{mid}-OP-{i:02d}',branch=branch,state='SCOPED_OPERATOR_ANALYSIS_WITH_EXPLICIT_REMAINING_PARAMETERS',evidence=evidence,gap=gap,scope=scope)
 common='Actualtensorgenerations/values/internalphasetimes/physicalcache andVectorTopKcapacity plusnewkernelreadiness/precision requirements unbound;implementationabsence alone isnotanunanalysedconditionalproposal.'
 m9=[
  row('M09',1,'C2unfiltered mode3 sourcepositions2/8/14,INT8QK/FP16headreduce/TopK;192rootgraphcalls.',common),
  row('M09',2,'C1candidate producer mode1 at20:positionTopKplus8-tokenblockamax/2048blockTopK,pinlatestvisibleblock;64rootcalls.',common+' Actualblockmembership/ties notcaptured.'),
  row('M09',3,'C1candidate consumer mode2 at24/28/32/36:fullCube scan thenmembershippenalty;outsidefinitepositions remainTopKfillers;256rootcalls.',common+' Finiteweightproduct/keyscale conditionforprefilterconstantproof unverified atcapturedvalues.'),
  row('M09',4,'Queryprojection→tailRoPE→perhead128INT8quant. QuantusesroundedFP16scale/div_rn/min2^-24/nearbyint;BF16ropeboundary retained.',common+' Allqueryprojectionshape-specificceilings stillrequireseparatework/weight/cachemodels.'),
  row('M09',5,'Weightprojection→BF16round→FP32scale→FP16round→FP16productwithqscale→Cubeweightedheadsum.',common+' Projection/scaleepilogueconsumer/resourcebounds notfullypriced.'),
  row('M09',6,'KeyNorm→BF16RoPE→DynamicQuant→keyScatter plusFP16scaleScatter;sourcequantizernotassumedqueryequivalent.',common+' Fullkeyquantizernumerics/slot-rowvaluegenerationpercall notattested.'),
  row('M09',7,'QLIfinalindices→visibility/chronologicalprepare→-1suffix/optionalvalidlength→stablebank. Currenthelperreturnssameprovidedbank;self-copyalias resolvedinsourcedomain.',common+' copy_/version/captureobservers notdiscardedbyassumption.'),
  row('M09',8,'Within-forwardsharedindicesandcandidatepublication;onecandidateproducerfeeds4consumers. Bitmap544B/query atcase34816 replacesmembershiplist8192B conditionally.',common+' Existingpublicationisbaseline;newbitmapbankneedscompletewrite/clear/generationandreadiness.'),
  row('M09',9,'Zeroquery/zerokey branches returnemptyor-1indices andzerooptional lengths;packed950backend excludedfromfixed910Ccase.',common+' Sourceguardsidentified;actualzero-branchcounts/otherunobservedapplicablepaths notmeasured.'),
  row('M09',10,'Crosscardkeypartition/localTopKmerge andcrossiterationdelta/reuse:ownrow/cardpartition andfullvaluegeneration equality required;query-change counterexample forbidsblanketreuse.',common+' Actualrankrowidentity/caches andnewcommunicationcosts notfullybound.'),
 ]
 m10=[
  row('M10',1,'SWA-only no compressedcache:window128,sinks,onlineSoftmax/PV;128rootgraphcalls.',common+' Actualvisiblewindow/validqueryrows notfrompoolshape.'),
  row('M10',2,'CSA C2:originalwindow plusselectedcompressedKV,page64observed,ratio2andcausalvisibility/indices;1152rootcalls.',common+' ActualvalidTopKcounts/union/sharedcachetier remainparameters.'),
  row('M10',3,'CSA C1:originalwindow plusselectedC1KV,page128observed,ratio1;1280rootcalls.',common+' Actualselectedcontents/cacheunion andtilepadding remainparameters.'),
  row('M10',4,'Draftoriginal-sparse branch:orisparsecapacity256 andperquerylengths;2/3queryshapes captured.',common+' Capacity256 doesnotprove256visiblekeys;window/mask/lengthvalues unrecorded.'),
  row('M10',5,'Noactualqueryrows→noSMLA,whileKV/cache-only workmayremain;priorv18cache-onlybranch evidence retained.',common+' HCorprojectioncountsdonotforceequalSMLAinvocationcount;allzerocallpopulations notattested.',e+['cache-only-KV-branch-reconciliation-v18.json']),
  row('M10',6,'AlreadyfusedSWA/CSAattention withFP32scoreGM→onlineSoftmax/sink→BF16expGM→PV/update/finalBF16;LSEdisabled0elements.',common+' Legalarch22handoff/capacity/actualGMcachetrafficnotproved;cannotannouncenoGMmerelysinglekernel.'),
  row('M10',7,'SMLA→64-tailinverseRoPE→send-layout/collective boundary;3200exactgraphchains.',common+' PreservetwoBF16rounds/448untoucheddims/validpadrows/allaliases/cardownership.'),
  row('M10',8,'Dependencyreorganization:SWAcaninprinciplestartbeforeIndexerselection;compressedbranchjoinneeds(m,l,a),sinkonce andoriginalnumericcontract.',common+' NewpartialstateABI/resourcebalance/extra launchandonlineorder conditional;BF16outputaloneinsufficient.'),
  row('M10',9,'Crosslayer/crossiterationreuse demandsquery,selectedkeyset,mask/positions,sink andKVgenerations identity;staticKVpoolalone doesnotallowreuseattentionresult.',common+' Otherunobservedapplicableoperatorbranchesandfullrequestcrosscardcriticalpath remainopen.'),
 ]
 write(O/'M09-M10-operator-branch-inventory-v60.json',dict(modules={'M09':m9,'M10':m10},
  scope='Semanticoperatorfamilies,not claimofallsourceguards orwholemoduleclosure. Priorchronologicalbranches archivedbypersistencehelper. NoM01/M21/M22/M25frameworkinventorychange.',overall_complete=False))
 print('6M models,13operatorvariants,19M09/M10semanticfamilies;case34816boundtohistoricalserveSHA;crossiterationcounterexample added.')

if __name__=='__main__':main()
