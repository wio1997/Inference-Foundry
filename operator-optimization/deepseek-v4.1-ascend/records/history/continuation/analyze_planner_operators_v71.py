"""Actual saved AICPU tasks plus source-bound planner algorithm, keys and publication costs."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib,math,re
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import fullkey,dimensions
from analyze_projection_ceiling_v61 import stat

def main():
    assert not(O/'planner-regime-models-v71.json').exists()
    previous=read(O/'operator-analysis-coverage-ledger-v70.json');pending={r['regime_id']:r for r in previous['regimes']if r['module_candidates']==['M21']and r['state'].startswith('REGIME_')}
    rs=[r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']if r['regime_id']in pending];assert(len(rs),sum(r['executions']for r in rs))==(21,608)
    lk={fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in rs};selected=defaultdict(dict)
    for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:
        if a[3]==['M21']and any(x in a[5]for x in ['(1174)','(1176)','(1254)']):selected[a[0]][a[1]]=a
    keys=defaultdict(list);times=defaultdict(list);missing=Counter();types=defaultdict(Counter)
    for di in range(32):
        e=read(CACHE/f'analysis/native-execution-identity-v14/db-{di:02d}.json');ts={t['task_record_rowid']:t for t in e['TASK_records']};ds={d['globalTaskId']:d for d in e['COMPUTE_definitions']}
        for tid,a in selected[di].items():
            t=ts[tid];d=ds[t['globalTaskId']];r=lk[fullkey(a[3],d['opType_text'],a[5],a[2],a[6],d)];rid=r['regime_id'];keys[rid].append([di,tid]);times[rid].append(t['endNs']-t['startNs']);missing[rid]+=not a[11];types[rid][t['task_type_name']]+=1
    src=read(O/'planner-operator-source-index-v71.json');api=read(O/'planner-API-source-binding-v71.json');assert src['all_requested_read']and len(src['files'])==11 and api['all2public_sources_match']
    files=src['files']+api['files'];assert len(files)==13
    for f in files:assert hashlib.sha256(Path(f['local_path']).read_bytes()).hexdigest()==f['sha256']
    assert sum(f.get('public_source_bytes_match',False)for f in files)==6
    get=lambda name:next(Path(x['local_path'])for x in files if x['path'].endswith(name))
    smla=get('/sparse_flash_mla_metadata_aicpu.cpp');qli=get('/quant_lightning_indexer_v2_metadata_aicpu.cpp');s=smla.read_text();q=qli.read_text()
    def body(text,marker):
        start=text.index(marker);lo=text.index('{',start);depth=0
        for i in range(lo,len(text)):
            depth+=(text[i]=='{')-(text[i]=='}')
            if depth==0:return text[lo:i+1]
        raise AssertionError(marker)
    sp=body(s,'::Prepare(');qp=body(q,'::Prepare(')
    attr=lambda txt:re.findall(r'GetAttrValue(?:Opt)?\(ctx,\s*"([^"]+)"',txt)
    sa,qa=attr(sp),attr(qp);assert len(sa)==len(set(sa))and len(qa)==len(set(qa))
    assert all(x in sa for x in ['soc_version','aic_core_num','aiv_core_num','is_batch_consistency'])
    assert all(x in qa for x in ['soc_version','aic_core_num','aiv_core_num','cmp_ratio'])and 'quant_mode'not in qa
    for text in [s,q]:
        assert '*metadataPtr = {};'in body(text,'::GenMetadata(')
        assert not any(t in text for t in ['getenv(', 'std::rand(', 'chrono::', 'uintptr_t'])
        assert body(text,'::BalanceSchedule(').count('CalcSplitPlan(')==1
    header_s=get('/sparse_flash_mla_kernel_metadata.h').read_text();header_q=get('/quant_lightning_indexer_v2_metadata.h').read_text()
    assert 'FA_METADATA_SIZE = 9'in header_s and 'FD_METADATA_SIZE = 8'in header_s
    assert 'QLI_V2_METADATA_SIZE = 8'in header_q and 'QLD_V2_METADATA_SIZE = 8'in header_q
    cfg=N/'sampling-guard-source-v55/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/config.json';assert hashlib.sha256(cfg.read_bytes()).hexdigest()=='43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7'
    text_cfg=read(cfg)['text_config'];assert {k:text_cfg[k]for k in ['num_attention_heads','head_dim','index_n_heads','index_head_dim','index_topk','sliding_window']}==dict(num_attention_heads=64,head_dim=512,index_n_heads=32,index_head_dim=128,index_topk=512,sliding_window=128)
    models=[]
    for r in rs:
        rid=r['regime_id'];assert len(keys[rid])==r['executions']and missing[rid]==r['missing_PMU'];op=r['operator'];d=r['signature'];a=dimensions(d['inputShapes_text']);z=dimensions(d['outputShapes_text']);assert z==[[1024]]and d['outputDataTypes_text']=='INT32'
        if op=='ZerosLike':
            assert sum(types[rid].values())==r['executions'];work=dict(family='CALLER_LOGICAL_ZERO_BANK',exact_value_stores=1024,known_value_write_bytes=4096,input_value_reads=0,native_empty_plan_is_different=True,conditions='SMLAcalleractualtokens0 fullzero/frontier. NativeallKVemptyGenMetadatastillcore0enable1,sozerobanknotinterchangeablewithnativeemptyplan. Elideonlynoquery/allreadersandpublicationeventconditions;otherwisefull1024orwholebankzero-invariant.')
        else:
            assert dict(types[rid])=={'KERNEL_AICPU':r['executions']}
            B=a[0][0]-1;assert B in [2,4];known=4*sum(math.prod(x)for x in a if x is not None)
            is_s=op=='SparseFlashMlaMetadata';defined=3600 if is_s else 3456;words=defined//4
            work=dict(family='SMLA_AICPU_PLAN'if is_s else'QLI_AICPU_PLAN',B=B,known_nonempty_input_descriptor_bytes=known,unknown_empty_descriptors_not_guessed=True,output_capacity_bytes=4096,defined_prefix_words=words,defined_prefix_bytes=defined,unwritten_suffix_bytes=4096-defined,source_whole_struct_zero_bytes=defined,semantic_store_reference_bytes='3888+28*U+24*V'if is_s else'3744+28*U (A3 V=0)',semantic_store_reference_conditions='A=24,Vcores=48,nonSplitG;UusedAIC1..24,VusedFD0..48. Sourcevaluestoremodelnotcompiledphysicaltraffic.',source_fixed_bank_copy_reference_bytes=8192,copy_TRACE_TASK_identity_proved=False,attribute_key=sa if is_s else qa,API_validation_key_must_also_retain='Allcallerattrs/types/layouts/presence/validations,includingQLIquant_modeevenifnotreadbythisAICPUPrepare.',row_cost_units='6*ceil(M/16)+10*ceil(S2/64);heuristicunitsNOTCPUcyclesorFLOPs',A3_row_blocking='mBase=Gifsparseorig/cmp,otherwisefloor(256/G)*G;s2Base512'if is_s else'mBase=4*(32/1)=128;s2Base2048;supportFd=falseonA3fresh/resetstate',cost_scan_row_count='R=sumceil(q_i*G/mBase)',fixed_case_row_scan_upper_bound=8192 if is_s else 2051,fixed_case_block_domain='Window128ororigTopK256⇒ori<=1block;cmpTopK512⇒cmp<=1block,soK<=2Runderthesecurrentbranches.'if is_s else'K<=17RundermaxModel34816;A3noFDskipsAssignByBlock,soKnotaclaimedblock-looptripcount.',schedule_stages=['ParamsCheck','ParamsInit','CalcSplitInfo','CalcBatchCost/CalcS1GCache','oneCalcSplitPlan(batch/row/optionalblockgreedy)','optionalSplitFD','GenMetadata'],input_value_not_shape_dependency=True,defined_plan_fields_contain_tensor_addresses=False,plan_purity_scope='Declaredfresh/fullyresetvalidinvocationandexactsource/effectiveattrs. Onlydefinedprefixdependsontensorvalues/presence/shapes+attrs,notpointeridentity. Currentcontainerfiles/opnamebinding,NOThistoricallyloadedELFattestation.',actual_input_values_and_hitrate_known=False)
        models.append(dict(regime_id=rid,module_candidates=['M21'],operator=op,role=r['role'],origin=r['origin'],context=r['context'],signature=d,executions=r['executions'],native_TASK_keys=keys[rid],native_TASK_ns=stat(times[rid]),native_TASK_types=dict(types[rid]),missing_PMU=missing[rid],work_model=work,consumer_fusion_review_complete_in_this_turn=True,physical_numeric_ceiling_proved=False))
    assert sum(missing.values())==529
    counts=Counter();caller=Counter()
    for r in models:counts[r['operator']]+=r['executions'];caller[(r['operator'],r['context'])]+=r['executions']
    assert counts=={'SparseFlashMlaMetadata':321,'VllmAscendQuantLightningIndexerV2Metadata':208,'ZerosLike':79}
    assert caller[('SparseFlashMlaMetadata','CALLER_DRAFT_OR_DRAFT_METADATA')]==69 and caller[('ZerosLike','CALLER_DRAFT_OR_DRAFT_METADATA')]==19
    support=[x for x in src['files']if 'prior_frozen_SHA_match'in x];assert len(support)==3 and all(x['prior_frozen_SHA_match']for x in support)
    write(O/'planner-regime-models-v71.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),regimes=models,regime_count=21,captured_execution_count=608,operator_execution_counts=dict(counts),scope='Explicitalgorithm/key/publication/saturation/layout/error/fusionconditionalmodelsfor21previouslypendingplanregimes. Actualscalarvalues/loadedELF/CPUcapacity/hitrateandallothermodelbranchesremainconditions.',overall_complete=False))
    write(O/'planner-source-algorithm-contract-v71.json',dict(source_files=files,SMLA_effective_AICPU_attrs=sa,QLI_effective_AICPU_attrs=qa,attribute_source='QLIexactoriginalop_apiinjectsplatformcube/vector/longSoC. SMLAoptionalcore/SoC/is_batch_consistencymustkeep_effectivevalues,notassumePythondefaults.',fresh_reset_state_condition=True,defined_schema=dict(SMLA=dict(FAmax=36,FAwords=9,FDmax=72,FDwords=8,defined_prefix_bytes=3600,unwritten_suffix_bytes=496),QLI=dict(AICmax=36,AICwords=8,AIVmax=72,AIVwords=8,defined_prefix_bytes=3456,unwritten_suffix_bytes=640)),no_address_payload_scope='GenMetadatavaluesareSplitResultintegerranges/relativeworkspaceindices/flags;inputpointersusedforloads,outputpointeronlystores. NoQ/K/cachepage/sparseindexactualvaluesinputs. SourcesnotproofloadedELF.',schedule='Onegreedypasspercorewithbatch→row→optionalblockandFDdistribution;noinventedbinarysearch. Blockcost6ceil(M/16)+10ceil(S2/64)integerheuristic;notattentionFLOPs.',full_source_key='Version+validattrs/presence/dtype/shape/layout+allreadlength/QSL/residual/topklengthvalues+SoC/AIC/AIV/is_batch_consistency+fresh/resetstate. Preservevalidationandbankfrontiers.',normal_form_key='FurthercoarseningonlyafterallusedSplitInfo/orderedrowCache/cost/FDflagsfieldsprovedidentical. Sameceil(S2/2048)aloneinsufficientbecauseceil(tail/64)changescost.',zero_branch='Callerfullzerobankdisablesallcores;nativeall-emptyKVplangenablesAIC0withbn2_end=B*Nkv. Notinterchangeable.',fixed_hardware_capacity='AICPUinteger/branch/cache-memory/launchthroughputnotknown;doNOTdivideplannerworkbyCube/Vectorpeakorautomaticallyby1.6TB/sHBM.',native_purity_all_actual_inputs_proved=False,loaded_native_binary_attested=False,new_remote_source_read=True,no_container_started_model_Torch_NPU_capture_health_process_kill_implementation=True,whole_modules_closed=0,overall_complete=False))
    write(O/'prior-support-file-binding-reconciliation-v71.json',dict(files=support,resolves_prior_current_filesystem_SHA_conditions=['v69forward_context.py','v70config/speculative.py','v70upstreamllm_base_proposer.py'],all3_exact_original_container_bytes_match_frozen=True,historical_loaded_code_identity_still_unproved=True,scope='Readexact3filesonlytobindpreviousoperatorguard/maskconclusions. No newframeworkanalysis;previousimmutablehistorykept.',overall_complete=False))
    write(O/'planner-source-local-validation-history-v71.json',dict(source_reads='11exactoriginalfiles+2QLIAPIexactfilesPASS;6core/APIfilesmatchofficialimmutablegitblobs;3priorguard/masksupportfilesmatchfrozenSHA.',resolved_error=dict(status='ERROR_RETAINED',stage='LocalAPIpublicblobcomparison',reason='apply_patchaddedoneextraterminatingLFto2publicAPIfiles;gitblobSHAassertcaughtitafterremotecopysucceeded. Noanalytical/sharedauthoritywritesoccurred.',recovery='RemovedexactoneLFonlywhenimmutableblobSHAverified;reusedsuccessfuloriginalremoteJSONpayloadwithoutrefetch. Oneunsuccessfultextpatchmadenochanges. CorrectNULseparatorwasnotcause.'),resolved_extractor_parse_error=dict(status='ERROR_RETAINED',reason='Missingwhitespaceafterelsebefore3456/2051/qaexpressions;Pythonparserpreventedexecutionandoutputwrites.',recovery='Correctedkeywordspacingandast.parsepassedbeforeactualsuccessful21/608extraction.'),model_extraction='21regimes608TASKidentitiesand529AICPUMissingPMUreconciled;nochangeglobal849.',no_model_Torch_NPU_capture_health_process_implementation=True,overall_complete=False))
    print(dict(regimes=21,executions=608,ops=dict(counts),missing_PMU=529,source_files=13,public_byte_matches=6,prior_support_matches=3,attrs=[len(sa),len(qa)],draft_split=[69,19],native_ns_by_operator={op:stat([v for r in models if r['operator']==op for v in times[r['regime_id']]])for op in counts}))

if __name__=='__main__':main()
