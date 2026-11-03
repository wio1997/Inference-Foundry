"""Explain a conclusion-changing profiler omission using pinned operator ABI text only."""
from pathlib import Path
from itertools import product
import ast, hashlib, json, re
from finalize_origin_coverage_v9 import N,O,W,read,write

def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def section(p,a,b):
    return dict(**pin(p),first_line=a,last_line=b,source=Path(p).read_text().splitlines()[a-1:b])

def report_prefix(args,cap=5):
    """Only the no-empty-shape domain needed for the non-injectivity witness."""
    result=[]
    for arg in args:
        if arg is None:continue
        name,kind,dtype,shape=arg
        assert shape  # No dependence on filtering/misalignment of scalar tensors.
        for role in (['input','output']if kind=='inout'else[kind]):
            if len(result)==cap:return result
            result.append(dict(name=name,role=role,dtype=dtype,shape=list(shape)))
    return result

def main():
    assert not(O/'REVIEW-CHECKPOINT-v117.json').exists()
    root=N/'sampling-launch-source-v117/usr/local/python3.12.13/lib/python3.12/site-packages/triton/backends/ascend'
    driver=root/'driver.py';utils=root/'utils.py';backend=root/'backend_register.py'
    pub=N/'sampling-launch-limit-v117/usr/local/Ascend/ascend-toolkit/latest/include/experiment/msprof/toolchain/aprof_pub.h'
    common=N/'sampling-launch-profiler-v117/usr/local/Ascend/ascend-toolkit/latest/include/experiment/msprof/toolchain/prof_common.h'
    api=common.with_name('prof_api.h')
    greedy=W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/ops/triton/reject_sample.py'
    sampler=greedy.parents[2]/'sample/rejection_sampler.py'
    hash_src=N/'state-engram-operator-source-v62/vllm_ascend/ops/triton/engram_hash.py'
    docs=[O/'heads-sampling-regime-models-v65.json',O/'sample-hash-profile-scope-v114.json',
        O/'sampling-operator-guard-contract-v102.json',O/'sampling-case-fusion-and-ceiling-routing-v102.json',
        O/'sampling-load-provenance-v102.json',O/'sampling-launch-source-index-v117.json',
        O/'sampling-launch-headers-index-v117.json',O/'sampling-launch-profiler-index-v117.json',O/'sampling-launch-limit-index-v117.json']
    files=[driver,utils,backend,pub,common,api,greedy,sampler,hash_src]+docs
    for receipt in ['sampling-profile-paths166-v117.json','sampling-launch-source166-v117.json',
        'sampling-launch-headers166-v117.json','sampling-launch-profiler166-v117.json','sampling-launch-limit166-v117.json']:
        files.append(N/receipt)
    write(O/'sampling-visibility-inputs-v117.json',dict(inputs=[pin(p)for p in files],
        questions=['Does omission of is_greedy BOOL prove all_greedy?',
            'Can tensor descriptors prove synthetic/random branches absent or hash output unwritten?',
            'Which positive per-call evidence would change operator-removal and ceiling scope?'],
        scope='Only operator argument observability; no framework survey, runtime imports, model or device work.',overall_complete=False))
    ptext=pub.read_text();cap=int(re.search(r'#define MSPROF_GE_TENSOR_DATA_NUM (\d+)',ptext)[1]);rank=int(re.search(r'#define MSPROF_GE_TENSOR_DATA_SHAPE_LEN (\d+)',ptext)[1]);assert(cap,rank)==(5,8)
    dt=driver.read_text()
    assert 'tensorCount < MSPROF_GE_TENSOR_DATA_NUM'in dt
    assert 'if idx < 5'in dt
    gnode=next(n for n in ast.parse(greedy.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='rejection_greedy_sample_triton')
    gargs=[a.arg for a in gnode.args.args]
    assert gargs[:6]==['output_token_ids_ptr','cu_num_draft_tokens_ptr','draft_token_ids_ptr','target_argmax_ptr','bonus_token_ids_ptr','is_greedy_ptr']
    hnode=next(n for n in ast.parse(hash_src.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_hash_ids_kernel')
    hargs=[a.arg for a in hnode.args.args]
    assert hargs[:5]==['input_ids','token_map','dead_mask','positions','block_table']and hargs.index('output')==13
    write(O/'sampling-visibility-source-contract-v117.json',dict(parts=[
        section(driver,702,736),section(driver,1193,1240),section(driver,1371,1378),
        section(driver,1539,1550),section(driver,1568,1588),section(driver,328,345),
        section(pub,30,36),section(pub,318,329),section(greedy,87,119),
        section(greedy,435,476),section(sampler,435,525),section(hash_src,46,83)],
        installed_pointer_descriptor_slots=cap,installed_shape_dimension_slots=rank,
        counting='One shared five-entry budget for input and output descriptions. INPUT_OUTPUT can consume two entries.',
        dtype_rule='Only first five pointer-signature dataTypes are generated; no proof of an unlimited descriptor channel.',
        shape_rule='Collect nonempty .size() results for pointer-typed signature entries in signature order; None/failed size/scalar empty shape do not produce a shape entry.',
        caution='This review uses no-empty-shape witnesses; it neither diagnoses a native ABI bug nor assumes missing metadata means absent readers.',
        metadata_kind_rule='Use packedMetadata tensor_kinds, cached by launcher; default input if kind unavailable. No current call metadata was recovered.',
        installed_source_vs_historical_loaded_launcher='UNKNOWN: fixed CANN acl_rt control matches, but no historical Triton launcher source/SO hash or call-to-specialization key was recorded here.',
        facts_grade='DOCUMENTED_INSTALLED_SOURCE;DERIVED_CONDITIONAL_REPORT_PROJECTION',
        no_target_module_import_or_device_work=True,overall_complete=False))
    regimes=[r for r in read(docs[0])['regimes']if r['operator']=='rejection_greedy_sample_triton'];assert len(regimes)==3
    cases=[]
    for r in regimes:
        w=r['work_model'];B=w['B'];T=w['draft_token_capacity'];S=w['max_speculative_tokens']
        prefix=[('output_token_ids_ptr','output','INT32',[B,S+1]),('cu_num_draft_tokens_ptr','input','INT32',[B]),
            ('draft_token_ids_ptr','input','INT32',[T]),('target_argmax_ptr','input','INT64',[T]),('bonus_token_ids_ptr','input','INT32',[B,1])]
        observed=report_prefix(prefix)
        assert ';'.join(x['dtype']for x in observed if x['role']=='input')==r['signature']['inputDataTypes_text']
        assert '"'+';'.join(','.join(map(str,x['shape']))for x in observed if x['role']=='input')+'"'==r['signature']['inputShapes_text']
        assert '"'+','.join(map(str,observed[0]['shape']))+'"'==r['signature']['outputShapes_text']
        variants=[]
        for mixed,synthetic in product([False,True],repeat=2):
            extra=[('is_greedy_ptr','input','DT_UINT1',[B])if mixed else None,
                ('uniform_probs_ptr','input','FLOAT',[T])if synthetic else None,
                ('synthetic_conditional_rates_ptr','input','FLOAT',[S])if synthetic else None]
            descriptors=report_prefix(prefix+extra);assert descriptors==observed
            variants.append(dict(nonempty_batch_domain='mixed_greedy_and_random'if mixed else'all_greedy',
                SYNTHETIC_MODE=synthetic,is_greedy_pointer_present=mixed,descriptors=descriptors))
        cases.append(dict(regime_id=r['regime_id'],executions=r['executions'],B=B,T=T,max_spec=S,signature=r['signature'],
            variants=variants,equal_projection=True,claim='Source-compatible indistinguishable descriptor witnesses, not reconstructed historical batches or values.'))
    assert sum(c['executions']for c in cases)==72
    hs=[r for r in read(docs[1])['existing_regimes']if r['operator']=='_hash_ids_kernel']
    assert sum(r['executions']for r in hs)==88
    write(O/'sampling-visibility-profile-projection-v117.json',dict(greedy_regimes=cases,
        greedy_executions_reused=72,projection_witnesses=len(cases)*4,
        hash_regimes=[dict(regime_id=r['regime_id'],executions=r['executions'],signature=r['signature'])for r in hs],
        hash_executions_reused=88,hash_tensor_argument_order=hargs[:14],hash_output_pointer_ordinal_one_based=14,
        hash_descriptors='Recorded first five inputs fit installed five-entry prefix. Output N/A is not evidence of no stores, no output readers or CPU execution.',
        runtime_all_greedy_or_synthetic_mode_recovered=False,new_native_promotions=0,overall_complete=False))
    # Boundary references verify that this is one shared budget, not five inputs plus five outputs.
    boundaries=[]
    for n in range(1,13):
        for io in range(n):
            args=[('a'+str(i),'inout'if i==io else'input','INT32',[i+1])for i in range(n)]
            got=report_prefix(args);assert len(got)==min(5,n+1)
            boundaries.append(dict(pointer_count=n,inout_index=io,description_count=len(got)))
    write(O/'sampling-visibility-proof-and-boundaries-v117.json',dict(boundaries=boundaries,
        noninjectivity='All-greedy/ordinary, mixed/ordinary, all-greedy/synthetic and mixed/synthetic share the same five descriptor prefix for each recorded geometry. This refutes inversion from these descriptor fields alone in the declared installed reporter model.',
        no_empty_scalar_or_DCE_premise_needed=True,
        missing_scalar_fields='all_greedy/all_random/SYNTHETIC_MODE and actual None specialization have no direct channel in these shape/type entries.',
        no_universal_historical_reporter_claim=True,overall_complete=False))
    write(O/'sampling-visibility-consequences-v117.json',dict(
        F35='Keep random/mixed/synthetic source branches conditional. Neither 72 greedy kernels nor four reported inputs closes an all-greedy guard. Preserve Fill -1 suffix and pre-return RNG/entropy/error/readers.',
        F24='Output N/A on hash task cannot remove the written hash-ID interface, its consumers or commit/rollback epochs; existing 48-INT32 IDs and v114 history factor contracts stay conditional.',
        costs='Do not turn optional-input omission into zero required work, zero RNG side effects, HBM-byte savings or a fixed-load physical speedup. Positive recorded descriptors remain usable under existing attribution gates.',
        historical_binding_required=['historical loaded launcher or source/SO digest','per-task compiled specialization key/constants','per-call all_greedy/all_random and request/metadata epoch','actual optional pointer presence plus pre-return entropy/synthetic effects'],
        acquisition='First reuse existing saved launcher/cache/log metadata if present and call-bound. An unrelated tiny operator probe cannot establish flags of the old 72 calls. No new model or device capture is authorized by this analysis.',
        remaining_analysis='Continue unexpanded operator relations and branches while actual guard attestation stays open. Do not repeat generic framework/MRO work or count these witnesses as new fusion opportunities.',
        new_pairs=0,new_groups=0,new_semantic_rows=0,new_native_regimes=0,new_native_executions=0,
        native_Pad_probe='PREPARED_NOT_EXECUTED;separate pending reply',new_device_work=False,
        physical_time_lower_bound_MFU_or_speedup_established=False,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',greedy_executions=72,hash_executions=88,descriptor_cap=cap,
        equivalent_projection_witnesses=12,boundary_references=len(boundaries),runtime_flags_recovered=False,new_device_work=False)))

if __name__=='__main__':main()
