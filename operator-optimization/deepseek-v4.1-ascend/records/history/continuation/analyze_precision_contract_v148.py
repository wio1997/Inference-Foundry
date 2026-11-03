"""Reconcile exported flags across every regime; gate actual instruction ceilings."""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
from itertools import product
import ast,hashlib,json,re,sqlite3
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
 assert not(O/'REVIEW-CHECKPOINT-v148.json').exists()
 pop=read(O/'precision-field-population-v148.json');dc={k:i for i,k in enumerate(pop['definition_columns'])};de={(r[0],r[1]):r for r in pop['definitions']};bytask={(r[0],r[1]):de[r[0],r[3]]for r in pop['executions']};keys=['blockNum','mixBlockNum','opType_text','inputShapes_text','inputDataTypes_text','outputShapes_text','outputDataTypes_text'];regs=read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes'];lk={(tuple(r['module_candidates']),r['operator'],r['role'],r['origin'],r['context'],tuple(r['signature'][k]for k in keys)):r for r in regs};assign_path=N/'original-profile-cache-v53/analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json';assign=read(assign_path);counts=defaultdict(Counter);proj=[];candidates=defaultdict(Counter);module=defaultdict(Counter)
 for a in assign['execution_assignments']:
  d=bytask[a[0],a[1]];sig=(d[10],d[11],d[5],d[8],d[6],d[9],d[7]);r=lk[tuple(a[3]),d[5],a[5],a[2],a[6],sig];flag=d[4];counts[r['regime_id']][flag]+=1;proj.append([a[0],a[1],r['regime_id'],flag]);candidates[tuple(a[3])][flag]+=1
  for m in a[3]:module[m][flag]+=1
 rr=[]
 for r in regs:
  c=counts[r['regime_id']];assert sum(c.values())==r['executions'];rr.append(dict(regime_id=r['regime_id'],module_candidates=r['module_candidates'],operator=r['operator'],role=r['role'],origin=r['origin'],context=r['context'],executions=r['executions'],exported_flag_counts=dict(c),actual_instruction_precision_or_native_operands_proved=False,existing_signature=r['signature']))
 mods=read(O/'module-profile-ledger-v147.json')['modules'];write(O/'precision-regime-module-projection-v148.json',dict(regimes=rr,regime_count=len(rr),execution_columns=['database_index','TASK_rowid','regime_id','decoded_flag'],execution_projection=proj,execution_count=len(proj),module_candidate_set_counts=[dict(module_candidates=list(k),flag_counts=dict(v))for k,v in sorted(candidates.items())],modules=[dict(id=m['id'],candidate_inclusive_flag_counts=dict(module[m['id']]),scope='CANDIDATE_INCLUSIVE_NOT_EXCLUSIVE_MFU_OR_RUNTIME_BRANCH_CLOSURE'if module[m['id']]else'NO_MATCHING_NATIVE_REGIME_IN_THIS_ATTRIBUTION;NOT_ZERO_UTILIZATION')for m in mods],all_prior_native_coverage_states_unchanged=True,no_new_primary_module_or_phase_attribution=True,new_device_work=False,overall_complete=False))
 root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
 for p,a,z in [(root/'ops/triton/spec_decode/dspark_swa_indices.py',8,45),(N/'HC-operator-source-v58/vllm-workspace/vllm-ascend/csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h',284,334),(N/'HC-operator-source-v58/vllm-workspace/vllm-ascend/csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h',418,477)]:parts.append(dict(**pin(p),first_line=a,last_line=z,source=p.read_text().splitlines()[a-1:z]))
 src=Path(parts[0]['path']).read_text();fn=next(x for x in ast.parse(src).body if isinstance(x,ast.FunctionDef)and x.name=='_build_dspark_swa_indices_kernel');calls=sorted({ast.unparse(x.func)for x in ast.walk(fn)if isinstance(x,ast.Call)});literals=sorted({type(x.value).__name__ for x in ast.walk(fn)if isinstance(x,ast.Constant)});integer_examples=[r for r in pop['definitions']if r[5]=='_build_dspark_swa_indices_kernel'];yeshost=Counter(r[12]for r in pop['definitions']if r[4]=='YES');matrix=[x for x in pop['op_dtype_flag_execution_counts']if 'MatMul'in x['opType']or'Matmul'in x['opType']or x['opType']in ['HcPre','DispatchFFNCombine']]
 unmatched_types=Counter()
 for ref in pop['databases']:
  db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True)
  local_strings={i:v for i,v in db.execute('SELECT id,value FROM STRING_IDS')}
  for ty,num in db.execute('SELECT t.taskType,COUNT(*) FROM TASK t LEFT JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId WHERE c.globalTaskId IS NULL GROUP BY t.taskType'):unmatched_types[local_strings.get(ty,'UNRESOLVED_DB'+str(ref['database_index'])+'_STRING_ID'+str(ty))]+=num
  db.close()
 docs=read(O/'precision-primary-receipts-v148.json');facts=[
  'MEASURED:32rawDBs/207750COMPUTE_TASK_INFOrecords/286950TASKjoinsdecodeviaeachDBSTRING_IDS,0missingflagstrings/0duplicateglobalTaskIddefinitiongroups.821262TASKswithoutCOMPUTEdefinitionareoutsideTHISfieldjoin;theyarenotautomaticallynewmissingarithmetickernelsorzerowork,andtheirprecisionisnotinferred. RawtaskTypecountsretained. COMPUTE namespace includesCOMMUNICATION-labeledrecords.',
  'DOCUMENTED_CONTEXT:CANN9.1beta1schemaexplicitlymaps hf32Eligible viaSTRING_IDS anddescribes YES/NOasuse/nonuseofHF32precisionflag,N/Aunrecognized. Exactinstalled9.2producer/fieldcoverageisnotattested. The integerforeignkeymustnotbecasttobool.',
  'MEASURED:All1120YESdefinitions/1120TASKjoinsareAI_VECTOR_CORE;explicitMatMulV2/V3/QuantBatchMatmulV3/TransposeBatchMatMul,HcPreandDispatchFFNCombineareNOinthispopulation. AYESflagisnotameasureofactualHF32MmadworkorpermissiontouseCubeHF32peak.',
  'SOURCE_SCOPED:ThefrozenDSparkindexkernelusesintegerload/compare/sum/max/index/store,withoutdot/floatconversion;sameopnamehas87YESand1NOrecords. Thisisnotbinaryhashidentityornewnativephaseproof. Arbitrarycompiler/sourceversionpossibilitiesremain;theflagalonecannotestablishmatrixinstructioncount.',
  'SOURCE_SCOPED:ReviewedHCABexplicitlySetHF32Mode(1),SetHF32TransMode(1),Mmad,SetHF32Mode(0). A2GramcallprecedestheABcallinselectedsourcepath,withfirstentrymodeunbound. WholeHcPreprofileNOdoesnotsettlewhichbodywasloadedorallinternalMmadstates;do notcalltheflagwrongorclaimactualNO/HF32contradictionresolved.',
  'DOCUMENTED_CONTEXT:LegacyCANN8.5boolSetHF32TransModetrueisRTZ,falseisnearest-even. Underthatbooloverload,C++literal1convertstrue;exact9.2SDKoverload/header/objectremainsunattested. CurrentmasterCAPIroundmodeenumusesseparatenames;do notmaplegacy1toanewenumordinal. Currentmasterfooter/hashisnotaninstalledSDKattestation.',
  'DERIVED_STATED_DOMAIN:Per-AICsourceABmodeautomatonON->Mmad->OFFdoesnotrestorepriorONstate;subsequentA2isOFFunderthatstraightlinepathuntilotherexplicitmodechange. Crosskernelcontextpersistenceandactualloadedinitialmodeunproven. Fusionthatmergescontextsmustpreserveoriginalphaseprecision/rounding,ratherthanuseonetasklabeloroneglobalmode.',
  'DERIVED_STATED_DOMAIN:Retain24restrictedclassicalHC/routerandC2modecaseswith24AIC2201reference2048/4096ops/cyclefromv89. Sharedsame-modepoolcutonlyunderactualmatchingmode/pool;different-modeindividualpeaksdonotproveweightedjointissuesimplex. Per-cyclemodelisnotphysicalclock/time/MFUorachievablespeedup.',
  'UNKNOWN_CRITICAL:ActualhistoricallyloadedHC/operatorbinary,mode/opcodecoverage/roundpolicy/initialstate,validphysicalfrequencyandcapacityupper,storage/networkcutsandsame-requestcausalendpointremainopen. No needtogenericallytraceframeworkexporter;returntooperatorfusion/readerworkwiththisprecisiongateexplicit.'
 ]
 write(O/'precision-field-source-and-primary-contract-v148.json',dict(parts=parts,documents=docs['documents'],facts=facts,index_kernel_AST_calls=calls,index_kernel_literal_types=literals,index_kernel_definition_flag_counts=dict(Counter(r[4]for r in integer_examples)),YES_host_type_counts=dict(yeshost),matrix_family_flag_execution_counts=matrix,unmatched_TASK_type_STRING_IDS_label_counts=dict(sorted(unmatched_types.items())),all_TASK_precision_fields_present=False,flag_decode_question_closed_in_full_saved_COMPUTE_namespace=True,actual_executed_instruction_mode_question_closed=False,whole_module_or_physical_MFU_closed=False,new_device_work=False,overall_complete=False))
 modecases=[]
 for m in [800,816,834]:
  wc=2*m*4*5120*24;wr=2*m*5120*384
  for c,r in product(['OFF','ON'],repeat=2):
   cp=24*(2048 if c=='OFF'else 4096);rp=24*(2048 if r=='OFF'else 4096);modecases.append(dict(kind='HC_COEFFICIENT_ROUTER',rows=m,HC_mode=c,router_mode=r,HC_work=wc,router_work=wr,HC_cycle_reference=wc/cp,router_cycle_reference=wr/rp,independent_cut_reference=max(wc/cp,wr/rp),same_mode_common_pool_cut_reference=(wc+wr)/cp if c==r else None,exported_flags='NO/NO;notautomaticinternalmodeattestation',native_mode_proved=False,actual_clock_or_time_lower_bound=None))
 for t in [6400,6528,6668]:
  w=2*t*5120*512
  for k,s in product(['OFF','ON'],repeat=2):
   kp=24*(2048 if k=='OFF'else 4096);sp=24*(2048 if s=='OFF'else 4096);modecases.append(dict(kind='C2_KV_SCORE',rows=t,KV_mode=k,score_mode=s,each_plane_work=w,KV_cycle_reference=w/kp,score_cycle_reference=w/sp,independent_cut_reference=max(w/kp,w/sp),same_mode_common_pool_cut_reference=2*w/kp if k==s else None,exported_flags='NO/NO;notautomaticinternalmodeattestation',native_mode_proved=False,actual_clock_or_time_lower_bound=None))
 automata=[]
 for initial,n in product(['OFF','ON'],[1,2,3,16]):
  state=initial;events=[]
  for j in range(n):events.append(dict(chunk=j,A2_mode=state,AB_mode='ON',after_AB='OFF'));state='OFF'
  automata.append(dict(initial_mode=initial,chunks=n,events=events,final_mode=state,loaded_binary_or_kernel_context_persistence_proved=False))
 write(O/'precision-gated-cycle-and-state-cases-v148.json',dict(cases=modecases,count=len(modecases),AIC_reference=24,OFF_ops_per_AIC_cycle=2048,ON_ops_per_AIC_cycle=4096,mode_automata=automata,scope='Restrictedclassicalwork+documented2201per-AICrateand24AICreference,notactualphysicaldenominator. Existingv89work/modesreusedwithnewprofilefieldgate,notanewHF32optimizationdiscovery.',new_numeric_physical_bound_or_speedup=False,new_device_work=False,overall_complete=False))
 witnesses=[]
 for mant in [10,11]:
  ulp=Fraction(1,2**mant)
  for sign in [-1,1]:
   for f in [Fraction(1,2),Fraction(3,4)]:
    x=sign*(1+f*ulp);rtz=sign*Fraction(1);rne=sign*(Fraction(1)+(ulp if f>Fraction(1,2)else 0));rna=sign*(1+ulp);witnesses.append(dict(mantissa_fraction_bits=mant,input=float(x),ulp=float(ulp),RTZ=float(rtz),nearest_even=float(rne),nearest_away=float(rna),actual_native_or_frozen_HC_values=False))
 rawbool_defs=sum(bool(r[3])for r in pop['definitions']);rawbyid={(r[0],r[1]):r[3]for r in pop['definitions']};rawbool_exec=sum(bool(rawbyid[r[0],r[3]])for r in pop['executions'])
 write(O/'precision-flag-rounding-inference-witnesses-v148.json',dict(rounding_cases=witnesses,wrong_bool_foreignkey=dict(enabled_definitions=rawbool_defs,actual_exported_YES_definitions=pop['definition_flag_counts']['YES'],enabled_TASK_joins=rawbool_exec,actual_exported_YES_TASK_joins=pop['execution_flag_counts']['YES'],method_legal=False),YES_not_matrix_work=dict(YES_host_type_counts=dict(yeshost),integer_index_source_calls=calls,instruction_work_inferred_from_flag=False),competing_HC_explanations=['LoadedbodymatchesfrozenABon/offwhileouterNOdoesnotinventoryinnerstates','Historicalbodydiffersfromreviewedsource;outerNOalonecannotselectthatbody','Exportedflagdenotesreportedflagcoverageanddoesnotattestperinstructiontrace'],loaded_explanation_selected=False,all_rounding_cases_finite_reference_not_native=True,new_device_work=False,overall_complete=False))
 names=['precision-field-population-v148.json','post-pre-precision-field-scope-audit-v147p.json','post-pre-postcheck-result-v147p.json','operator-shape-ceiling-ledger-v55.json','operator-analysis-coverage-ledger-v147.json','module-profile-ledger-v147.json','post-pre-native-frontiers-v147.json','hc-norm-projection-native-frontiers-v143.json','typed-capacity-primary-contract-v89.json','HC-precision-mode-capacity-cases-v89.json','precision-primary-receipts-v148.json'];inputs=[pin(O/n)for n in names]+[pin(assign_path)]+[pin(Path(x['path']))for x in docs['documents']]
 write(O/'precision-field-inputs-v148.json',dict(inputs=inputs,profile_first=True,questions=['Canrawhf32EligibleintegersbeusedasBooleanorarithmeticmode?','DoexportedYES/NOflags inventorytheinternalHC/router/C2Mmadprecisions?','Whichrestrictedcyclecasesremainandwhatactualbinary/state/capacityevidenceisstillneeded?'],source_scope='Threeboundedparts:oneintegerDSparkindexkernelandtwoHCcubeMmadcallregions;officialschema/HF32/legacyboolAPIonly. No framework-exporter/SDKsurvey/deviceprobe.',new_device_work=False,overall_complete=False))
 print(json.dumps(dict(status='PRECISION_FLAG_SCOPE_RECONCILED',regimes=len(rr),TASK_joins=len(proj),modules=len(mods),YES_TASKs=pop['execution_flag_counts']['YES'],mode_cases=len(modecases),mode_automata=len(automata),new_pair_clauses=0,new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False)))


if __name__=='__main__':main()
