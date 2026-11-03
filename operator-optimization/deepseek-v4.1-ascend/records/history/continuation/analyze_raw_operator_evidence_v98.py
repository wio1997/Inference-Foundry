"""Reconcile saved raw operator attributes with exact cached execution identities; no device work."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json
from finalize_origin_coverage_v9 import N,O,R,read,write
from mirror_saved_profile_core_v53 import CACHE

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raw_key(r):return tuple(r[k]for k in ['attrInfo','input_shapes','output_shapes','input_dtypes','output_dtypes'])
def definition_key(r):return tuple(r[k]for k in ['attrInfo','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text'])

def main():
    assert not(O/'REVIEW-CHECKPOINT-v98.json').exists()
    packet=read(O/'raw-operator-evidence-extraction-v98.json');raw=packet['result']
    assert packet['returncode']==0 and not raw['new_capture'] and len(raw['databases'])==32
    cap=read(CACHE/'analysis/profile-capabilities-index-v12-v2.json')
    configs=json.loads(read(N/'saved-metadata-config166-v53.json')['stdout'])['capture_configurations']
    assert len(configs)==32
    info={r['path']:r for r in cap['profiler_infos']}
    conf=[]
    for r in configs:
        assert info[r['original_path']]['sha256']==r['sha256']
        conf.append({k:r[k]for k in ['original_path','sha256','rank_id','torch_npu_version','cann_version','common_config','experimental_config']})
    signatures=Counter(json.dumps({k:r[k]for k in ['torch_npu_version','cann_version','common_config','experimental_config']},sort_keys=True)for r in conf)
    schema=Counter(json.dumps(db['all_table_columns'],sort_keys=True)for db in raw['databases'])
    assert len(schema)==1 and all(not db['candidate_address_or_argument_columns']for db in raw['databases'])
    totals=defaultdict(Counter)
    for db in raw['databases']:
        ref=cap['databases'][db['database_index']]
        assert (db['path'],db['sha256'],db['bytes'])==(ref['path'],ref['sha256'],ref['bytes'])and db['stat_unchanged']
        for row in db['operators']:totals[row['operator']].update(executions=row['executions'],with_nonempty_attr=row['with_nonempty_attr'])
    write(O/'raw-profile-capability-audit-v98.json',dict(
      capture_configurations=conf,configuration_signature_counts=[dict(config=json.loads(k),count=v)for k,v in signatures.items()],
      same_hash_databases=32,raw_database_bytes=raw['bytes_read_for_identity'],schema_variants=len(schema),tables_per_database=27,
      raw_tables_and_columns=raw['databases'][0]['all_table_columns'],candidate_address_or_argument_columns={},operator_attribute_census=dict(totals),
      interpretation=[
        'All32saved profiler configurations disable record_op_args,_op_attr,profile_memory andwith_flops;record_shapes/with_stack/with_modules are enabled. This does not imply every COMPUTE_TASK_INFO.attrInfo is empty.',
        'NonemptyattrInfo is attached to selected compute definitions,including3200MC2executions. Decode own-op attributes,not neighboring operators or historical hashes.',
        'These27table schemas contain no operand-address/length/workspace/allocation-lifetime columns. ENUM_MEMCPY_OPERATION is an enum,not recorded memcpy operands. Absence is bounded to these32hash-pinned exported DBs,not every possible saved file or underlying runtime.',
        'TASK_PMU_INFO exists and must not be erased by profiler flags. No new PMU completeness,physical rate or universal ceiling has been established.'
      ],new_capture=False,server_scope='Readonly existing pinned files;no model/Torch/NPU/service/source change.',overall_complete=False))

    rows=[];joined_groups=0;perdb=[]
    for db in raw['databases']:
        i=db['database_index'];p=CACHE/f'analysis/native-execution-identity-v14/db-{i:02d}.json';cache=read(p);cache_sha=digest(p)
        assert cache['database_sha256']==db['sha256']
        defs={d['globalTaskId']:d for d in cache['COMPUTE_definitions']if d['opType_text']=='DispatchFFNCombine'}
        groups={raw_key(g):g for g in db['target_operator_regimes']['DispatchFFNCombine']}
        assert len(groups)==len(db['target_operator_regimes']['DispatchFFNCombine'])
        observed=Counter();counters=Counter()
        for t in cache['TASK_records']:
            if t['globalTaskId']not in defs:continue
            d=defs[t['globalTaskId']];key=definition_key(d);g=groups[key];observed[key]+=1
            a=dict(x.split(':',1)for x in g['attrs'].split('|'))if g['attrs']else{}
            if a:
                assert a=={'attr_0':'group_name_137','attr_1':'65536','attr_2':'0','attr_3':'1','attr_4':'10.000000','attr_5':'16'}
            C=int(a['attr_1'])if a else None;counters['executions']+=1;counters['recorded_capacity']+=int(C is not None)
            rows.append(dict(database_index=i,database_sha256=db['sha256'],cache_path=str(p),cache_sha256=cache_sha,
              compute_record_rowid=d['compute_record_rowid'],globalTaskId=d['globalTaskId'],task_record_rowid=t['task_record_rowid'],
              startNs=t['startNs'],endNs=t['endNs'],streamId=t['streamId'],modelId=t['modelId'],deviceId=t['deviceId'],
              raw_attribute_reference_id=d['attrInfo'],raw_attribute_text=g['attrs'],input_shapes=d['inputShapes_text'],
              recorded_capacity=C,recorded_world_size=int(a['attr_5'])if a else None,
              recorded_is_trans_b=bool(int(a['attr_2']))if a else None,recorded_weight_nz=bool(int(a['attr_3']))if a else None,
              recorded_swiglu_limit=float(a['attr_4'])if a else None,
              source_index_interpretation='Frozen customMC2 host attr indices1capacity/2transB/3weightNZ/4swigluLimit/5worldSize;not loaded-machine-code attestation.',
              all_rank_same_call_input_bound_or_receiver_count=None,actual_workspace_allocation_or_time_credit=None))
        assert observed==Counter({k:g['executions']for k,g in groups.items()})
        joined_groups+=len(groups);perdb.append(dict(database_index=i,cache_sha256=cache_sha,**counters))
    assert len(rows)==4472 and sum(r['recorded_capacity']is not None for r in rows)==3200
    known={(r['database_index'],r['task_record_rowid'])for r in rows if r['recorded_capacity']is not None}
    front=read(O/'MC2-operator-profile-v56.json')
    bound={(r['database_index'],r['route_task_rowid'])for r in front['root_bound_graph_layers']}
    unbound={(r['database_index'],r['route_task_rowid'])for r in front['phase_unbound_graph_layers']}
    assert known==bound|unbound and not bound&unbound
    host=N/'MC2-operator-source-v56/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp'
    lines=host.read_text().splitlines()
    write(O/'MC2-recorded-attributes-and-executions-v98.json',dict(executions=rows,per_database=perdb,joined_raw_groups=joined_groups,
      recorded_capacity_executions=3200,capacity_missing_executions=1272,recorded_capacity=65536,recorded_world_size=16,
      recorded_target_shape='M3,H5120,k6,localE24',root_bound_executions=len(bound),phase_unbound_executions=len(unbound),
      source=dict(path=str(host),sha256=digest(host),segments=[dict(first_line=a,last_line=b,source=lines[a-1:b])for a,b in [(31,43),(88,119)]]),
      prior_projection_missing_is_not_raw_absence=True,unknown_capacity_not_filled_by_default=True,
      source_workspace_formula_with_recorded_C_reference_bytes=1023993344,
      source_formula_and_loaded_ABI_binding_still_conditional=True,actual_allocator_peak_bytes=None,actual_receiver_count=None,
      no_time_or_HBM_credit_from_capacity=True,overall_complete=False))

    init=read(O/'initialization-profile-insufficiency-v67.json');router=read(O/'router-MemSet-profile-insufficiency-v63.json');recorded=[]
    rawdb={d['database_index']:d for d in raw['databases']}
    for r in init['records']+router['native_records']:
        d=r['definition'];t=r['TASK'];i=r['database_index'];op=d['opType_text']
        group=[g for g in rawdb[i]['target_operator_regimes'][op]if raw_key(g)==definition_key(d)]
        assert len(group)==1
        recorded.append(dict(database_index=i,task_record_rowid=t['task_record_rowid'],operator=op,raw_attribute_text=group[0]['attrs'],
          input_shapes=d['inputShapes_text'],input_dtypes=d['inputDataTypes_text'],destination_address=None,native_write_length=None,
          semantic_field_or_all_consumers=None))
    pending=read(O/'whole-scope-requirement-audit-v97.json')['pending_native_rows']
    assert len(recorded)==1352 and sum(r['operator']=='MemSet'for r in recorded)==1272
    assert all(r['raw_attribute_text']==''for r in recorded if r['operator']=='MemSet')
    assert all(r['raw_attribute_text']=='IsStaticKernel:false'for r in recorded if r['operator']=='ZerosLike')
    write(O/'remaining-operand-evidence-status-v98.json',dict(pending_regimes=pending,execution_bindings=recorded,
      MemSet=dict(executions=1272,regimes=7,raw_own_attributes_empty=True,shapes_and_dtypes='N/A',pointer_or_length_found=False,
        conclusion='ParentConstantPadNd bound remains useful,but cannot prove thisnativehelperwriteextent,overlap,zero-elisionorfusionlegality.'),
      BOOL_ZerosLike=dict(executions=80,regimes=2,shape=[24],raw_attributes='IsStaticKernel:false',
        semantic_field_and_all_consumers_proved=False,conclusion='Kernelstaticness is not mask/fieldidentity. Do not inferfalsepadding,deadinitializationorrowpruning.'),
      all9remain_pending=True,missing_fields_not_zero_work=True,overall_complete=False))

    dispatch=N/'MC2-operator-source-v56/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp'
    ls=dispatch.read_text().splitlines()
    write(O/'MC2-attribute-to-ceiling-contract-v98.json',dict(
      profile_recorded_capacity=65536,profile_executions=3200,local_input_rows=3,local_topK=6,local_experts=24,
      demand='a_e is same-call preclip demand for localexpert e;A=sum_e a_e. Source processeseachgroup withr_e=min(a_e,max(0,C-sum_previous a_j));sum_e r_e=min(A,C).',
      classical_dense_GEMM_reference='F=2*sum_e r_e*(5120*4608+2304*5120)=70778880*min(A,C) arithmeticOPS for thetwoINT8denseGEMMs. Not vector/quant/communication work;not a universal all-algorithm lower bound.',
      reference_OPS_per_processed_assignment=70778880,
      current_dense_work_or_MFU=None,
      profile_comparison=dict(recorded_M3_TASK_duration_ns=next(r['duration_ns']for r in front['shape_timings']if r['executions']==3200),
        conditional_int8_peak_reference_fraction='70778880*min(A,C)/(P_INT8_upper*T_TASK_seconds);samephysicalscope/OPSprecision/ratecontractrequired. UnknownA => unknownfraction;not all-moduleMFU.',
        native_TASK_span_not_end_to_end_request_critical_path=True),
      lower_bound_contract='Only provennecessaryworkinthedeclaredalgorithm/precisiondomain dividedbyvalidcapacityuppercanenterL. WholefusedpathL>=max(L_matrix,L_vector,L_memory,L_network,L_dependencies) withshared-resourceaccountingandtile-streaming. ProfileCisnotA,GMallocationisnotHBMbytes,andfusedstandalonelowerboundssummustnotdoublecount.',
      guards=['BOOL[3]x_active_mask values areabsent;M3isallocatedrowgeometry,notproofall3active.',
        'recordedworldSize16 andidenticalgroupname text do notestablishsame-call all-rank rowbounds orjointreceivercounts.',
        'ChangingCwhilepreservingclippingdoesnotbyitselfdeleteclassicalsemanticGEMMwork;allocationrequestchangecannotbeconvertedtoFLOPs/HBMtime.',
        'Source expert_token_nums isperexpert preclipdemandafterEPprefixreduction,notanexpertwise cumulativelist;sumonlywithloadedsource/ABI/integerdomainbound.'],
      source=dict(path=str(dispatch),sha256=digest(dispatch),segments=[dict(first_line=a,last_line=b,source=ls[a-1:b])for a,b in [(401,433),(451,477),(516,518),(558,580),(620,622),(821,868)]]),
      numerical_physical_ceiling_or_critical_gain_proved=False,overall_complete=False))
    write(O/'minimal-operator-evidence-plan-v98.json',dict(status='PREPARED_NOT_EXECUTED',source_scope='Only evidence that changes operatorfusion/split/reorganization legality,necessarywork orresourceceilings.',
      no_new_capture_or_target_change_authorized_by_this_plan=True,
      optional_next_windows=[
        dict(id='SAMECALL_RECEIVER',question='Bind source-row/topK bounds or receiver demand to the exactcustomMC2execution with recordedC.',
          fields=['run/causal communication-instance andgroup/rank identity','exactrecordedC,M,k,E andloadedkernel/path identity','same-call expert_token_nums anditsdtype/shape','workspace source-request vsactualallocation/capturebankgeneration ifclaimingmemorycredit'],
          bound='One controlled completeEPgroup window;at mostone selectedlayer andone callperrank perworkloadcohort,16rankreference. Nofullactivation orweightdump.',
          optional_device_copy='24INT32counts=96B/rank target,8INT32=32B/rank draft. New NPU-to-CPU read/synchronization requires an authorized finite diagnostic window;latency is perturbed and cannot be used asbaseline.',
          source_count_semantics='GetCumsumForMMAIV accumulates acrossEPsources independentlyforeachexpert. LastsourcevectorcopiedtoExpertTokenNums thereforehasper-local-expert demandbeforemaxOutputSizeclipping;sumofvalidnonnegativeentriesisreceiverdemandinthissourcecontract,notanexpert-prefixsum. Loadedbinaryandinteger-rangedomainstillmustbind.',
          source=dict(path=str(dispatch),sha256=digest(dispatch),segments=[dict(first_line=a,last_line=b,source=ls[a-1:b])for a,b in [(401,433),(821,868)]]),
          sufficiency='Observedreceiverdemand alone doesnotprove allfuturecase safety. Use a boundedsamecall proof forcapturedcase orprovable workloadmax for reusable sizing. sourcebinary/writetile/alias/capturegates remain.'),
        dict(id='MEMSET_OPERANDS',question='Which destination bytes are initialized,andwhichproducer/consumer observes them?',
          fields=['parentoperator andexactTASK/callidentity','destinationaddress+length+fillvalue','allocation/aliasgeneration','alloverlappingwritersandreaders andreadiness'],
          bound='Separate the3M12 and4M19regimes;one representativeperregime plusexplicituniformitycontract. Asteadydecode-onlywindow cannotcoverdummy/prefillbranches.',
          mechanism='Preferalready-savedIR/argumentmetadata ifindexed. Otherwise prepare boundednativeargument/IR diagnostic;PythoncallstackorCPUshapes alone cannot supplynativeMemSet destination/length.',
          sufficiency='Anaddress/lengthsample doesnotcloseallcase writeextents orallreaders;noautomaticzero-elision.'),
        dict(id='BOOL_FIELD_READERS',question='What exactBOOL[24]field iszeroed,anddo allreaders permitfusedinitialization?',
          fields=['operatorgraphvalue/aliasidentity','fieldproducer andallconsumers','row/paddingdomain andepoch','materializedsizeandnativewritegranularity'],
          bound='Bothphase-unbound16 anddecode-compatible64executionregimes;staticnessattributealone isinsufficient.',
          mechanism='Targetedgraph/operandidentity evidence;noframework-widesurvey.')],
      stop_conditions=['No availableauthorizedfinitecapturewindow:keepplanpreparedandcontinueindependentoperatoranalysis.',
        'Identity/hash/causalbindingfails:do notmergeotherhost/capture/rankordinalrecords.',
        'Anynewdiagnosticmustretainunprofiledbaselineandmarkinstrumentationperturbation;nofusionimplementationorconfigurationchange.'],
      quantitative_scope=dict(new_native_executions=0,new_capture=0,device_to_host_copies=0),overall_complete=False))
    print(json.dumps(dict(raw_databases=32,MC2recordedC=3200,MC2unknownC=1272,root_bound=len(bound),phase_unbound=len(unbound),pending_operand_executions=len(recorded),overall_complete=False)))

if __name__=='__main__':main()
