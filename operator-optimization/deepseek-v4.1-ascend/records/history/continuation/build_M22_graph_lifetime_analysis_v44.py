"""Saved-profile placement derivation and finite graph lifetime/source-scope audit."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import hashlib
import json

from finalize_origin_coverage_v9 import N,O,W,read,write
from finalize_Engram_whole_frontier_v30 import packet,graph


def old_pin(version,name):
    entries=read(O/f'checkpoint-manifest-v{version}.json')['artifacts']
    a=next(a for a in entries if Path(a['path']).name==name)
    b=Path(a['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==a['sha256']
    return dict(version=version,path=a['path'],sha256=a['sha256'])


def main():
    assert not(O/'REVIEW-CHECKPOINT-v44.json').exists()
    stamp=datetime.now(timezone.utc).isoformat()
    source=read(O/'M22-capture-lifetime-point-source-v44.json')
    runner=read(O/'M22-runner-update-and-enpu-binding-point-source-v44.json')
    for x in [source,runner]:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
    named=packet('M21-native166-v37')['summary']
    global_graph=packet('saved-graph-resources166-v14')
    assert named['typed_metadata_without_native_fullcaller_TASKs']==0
    assert named['metadata_type_counts']=={'SparseFlashMlaMetadata':321,'VllmAscendQuantLightningIndexerV2Metadata':208}
    resource=read(O/'resource-coverage-summary-v20.json')
    assert resource['capture_origin_partition']==dict(DIRECT=154940,GRAPH=132000,QUEUE=10)
    pins=[old_pin(37,'query_M21_native_metadata_v37_166.py'),old_pin(20,'aggregate_resource_roles_v20_166_v3.py')]
    # Verify the prior complete queries use exhaustive typed fallback and disjoint identity sets.
    q=(N/'query_M21_native_metadata_v37_166.py').read_text()
    a=(N/'aggregate_resource_roles_v20_166_v3.py').read_text()
    assert "for t in d['TASK_records']"in q and 'typed_only.append'in q
    assert "assert key not in ginfo"in a and "assert len(chain_by_key)==154940"in a
    placement=dict(
        generated_utc=stamp,grade='DERIVED_FROM_PINNED_COMPLETE_CENSUSES_AND_IDENTITY_CLASSIFIER',
        named_metadata_TASKs=named['metadata_type_counts'],typed_metadata_without_native_fullcaller=0,
        exhaustive_SCAN='v37 scans every TASK_record of all32 pinned E/index parts; each metadata-type record is native-linked or retained typed_only. typed_only0 and named counts321+208 exhaust these two named types.',
        disjointness='v20 classifier covers all154940 source-context identities and asserts each DIRECT key absent from graph identity set. Same native E/index/chain definitions underlie v37.',
        derived_named_metadata_origins={'SparseFlashMlaMetadata':dict(DIRECT=321,GRAPH=0),'VllmAscendQuantLightningIndexerV2Metadata':dict(DIRECT=208,GRAPH=0)},
        global_capture_origin_partition=resource['capture_origin_partition'],historical_immutable_query_pins=pins,
        evidence=['M21-native166-v37.json','saved-graph-resources166-v14.json','resource-coverage-summary-v20.json'],
        fresh_remote_identity_join=False,
        limits=['Only these two named metadata ComputeInfo TASK types; not absence of opaque/internal/fused plan work or every tensor store.',
                'DIRECT placement is not startup/formal/request attribution, pointer equality, consumer capture ABI, purity, no-effect or value-generation reuse proof.',
                '529metadata missingPMU remain missing; no costs/savings inferred.'],
        whole_module_closed=False,overall_complete=False)
    write(O/'M22-named-metadata-graph-placement-derivation-v44.json',placement)

    facts=[
        dict(grade='DOCUMENTED',statement='Wrapper constructor forbids NONE, takes global graph pool, optional graph options, independent enable_super_kernel, enable_enpu/use_eagle and update_stream, initializes backend None and descriptor-keyed entries. These fields are not all implied by saved enable_npugraph_ex or enable_static_kernel.'),
        dict(grade='DOCUMENTED',statement='Runtime NONE/mode mismatch runs callable directly; matching mode creates descriptor entry and captures only if entry.aclgraph is None. Capture uses args/kwargs current values; replay reuses graph and returns stored output rather than re-evaluating runtime args/kwargs.'),
        dict(grade='DOCUMENTED',statement='First capture records addresses only for top-level positional Tensor args. Debug replay checks the same restricted list. kwargs and nested Tensor containers are not enumerated by this check.'),
        dict(grade='DERIVED_FROM_PINNED_SOURCE',statement='NPUModelRunner._model_forward builds model_inputs dict and partial(self.model,**model_inputs), then calls run_model() with no positional inputs. For this ordinary wrapper call path, captured/debug positional-address lists can both be empty; passing the address assertion does not validate input_ids/positions/embeds/metadata banks.'),
        dict(grade='DOCUMENTED',statement='Before capture, offloader sync_prev_onload completes prior prefetch; join_after_forward joins its copy stream within capture. gc/empty_cache monkeypatch lifetime is scoped by ExitStack. RuntimeError diagnostic branches rethrow, preserving exception chains; no actual failure/upgrade recommendation is inferred.'),
        dict(grade='DOCUMENTED',statement='Optional weak_ref_output may release strong output during capture; current source says it is safe only for last piecewise graph. Workspaces and entry.output are then weak-ref converted. First capture returns output, later replay returns stored output. This describes references/lifetime, not a device-copy or fresh independent result allocation.'),
        dict(grade='DOCUMENTED',statement='Main ENPU flag comes from C getenv ENPU_ENABLE, true only for a non-None case-insensitive true string; enable_npugraph_ex is a different configuration field. No saved ENPU_ENABLE value is attested here.'),
        dict(grade='DOCUMENTED',statement='Wrapper non-ENPU FULL/non-draft-EAGLE replay synchronizes current stream before replay. Updatable path resolves current ContextSource or draft SharedSource; ENPU branch updates then replays, non-ENPU replays then updates. Native parameter-epoch semantics cannot be changed from Python order alone.'),
        dict(grade='DOCUMENTED_FROZEN_PREDICATE',statement='Local utils use_updatable_graph requires non-None backend subclass of AscendAttentionBackend. It is not synonymous with every graph/Deepseek backend. The utils hash was computed locally; this turn does not attest its deployed166 hash or live backend class.'),
        dict(grade='DOCUMENTED',statement='Runner generic update_full_graph_params helper is guarded FULL and not capturing and not use_sparse and not use_compress; its ENPU branch also synchronizes. use_compress is hf_config-present+compress_ratios attribute, use_sparse=enable_sfa(vllm_config). Kernel name alone does not bind these live Python flags.'),
        dict(grade='DOCUMENTED',statement='Runner calls generic update before model for ENPU, after model otherwise. In finally it releases an in-flight DeviceMetadataExecutor submission even if forward raises, preventing later submit deadlock. Preserve this exception path in any generation-pipeline rewrite.'),
        dict(grade='DOCUMENTED_PLUS_PRIOR_REVIEW',statement='Engram routing is prepared before each model replay; capture path uses graph input banks, noncapture path uses current IDs/positions/lookback and device inputs. Existing v30 cache/history/DP participation proofs remain; graph reuse does not make data-dependent routing reusable across iterations.'),
        dict(grade='DOCUMENTED_PLUS_PRIOR_REVIEW',statement='SMLA plan producer arguments include CPU-derived integer max lengths/request/head/layout fields as well as device length tensors (v37/v42 producer contract). Moving the call into a graph must preserve the scalar argument generation or a separately proved safe fixed bound; updating only GPU tensor contents is insufficient as a generic proof.'),
    ]
    write(O/'M22-capture-replay-input-output-and-update-contract-v44.json',dict(
        generated_utc=stamp,facts=facts,source=['M22-capture-lifetime-point-source-v44.json','M22-runner-update-and-enpu-binding-point-source-v44.json'],
        fixed_saved_main_settings='v42 DP2/TP8,DCP1,main capture sizes[8,16,24]/warmup1/copy_inputsFalse; draft outer eager handled separately.',
        actual_ENPU_ENABLE=None,actual_wrapper_backend_class=None,actual_breakable_mode=None,
        remaining=['Actual per-descriptor/bank/graph identities and all first/last readers/alias/reuse epochs.',
                   'ENPU_ENABLE, backend/update-task eligibility, breakable/super-kernel modes where decisive.',
                   'Native update parameter epochs and actual metadata memory contents;16phase-unbound graphs.',
                   'Original worker/capture/registry/weak-workspace adjacent scope beyond bounded41 guards.'],
        implementation=False,native_validation=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))

    # Concrete limitation of the source address check, not a runtime Tensor test.
    def checked(args):return [x['address']for x in args if isinstance(x,dict)and x.get('kind')=='tensor']
    old_args=[];new_args=[];old_kwargs={'input_ids':{'kind':'tensor','address':10}};new_kwargs={'input_ids':{'kind':'tensor','address':20}}
    assert checked(old_args)==checked(new_args)==[] and old_kwargs!=new_kwargs
    old_nested=[({'kind':'tensor','address':30},)];new_nested=[({'kind':'tensor','address':40},)]
    assert checked(old_nested)==checked(new_nested)==[]
    write(O/'M22-positional-address-check-domain-proof-v44.json',dict(
        generated_utc=stamp,grade='DERIVED',kwargs_counterexample=dict(old=old_kwargs,new=new_kwargs,old_checked=checked(old_args),new_checked=checked(new_args)),
        nested_counterexample=dict(old=old_nested,new=new_nested,checked_equal=True),
        source_call_path='partial(self.model,**model_inputs) then run_model() has no positional model inputs.',
        implication='Debug positional address equality is insufficient as all-input fixed-address evidence for fusion or bank replacement. External owners must preserve captured kwargs/nested tensors and all producer/readiness/lifetime contracts.',
        no_actual_pointer_values=True,no_target_change=True,whole_module_closed=False,overall_complete=False))

    # Preserve original scope without treating1101 static records as analyzed graph branches.
    orig=next(x for x in read(W/'module-coverage-ledger.json')['surfaces']if x['id']=='M22')
    indexed=read(W/'coverage-ledger-v3/branch-sites.json');ids=set(orig['indexed_branch_ids']);sites=[deepcopy(x)for x in indexed if x['id']in ids]
    assert len(sites)==len(ids)==1101
    frozen=W/'frozen-source-snapshot';hashes={s['source']:s['source_sha256']for s in sites}
    for rel,sha in hashes.items():assert hashlib.sha256((frozen/rel).read_bytes()).hexdigest()==sha
    definitions=[
        ('S01','Wrapper dispatch/capture/replay and entry/pool/options','LOCAL_CONDITIONAL_CORE_ANALYZED','Descriptor,mode,capture,input/debug/output/ref/offload/cache/exception branches.','Actual graph/bank generations and external option/backend binding remain.'),
        ('S02','Graph parameter registry, weak workspaces and proxy/diagnostics','FINITE_REGISTRY_WORKSPACE_CROSSWALK_PENDING','Remaining graph-file helpers and parameter/workspace stores.','28 other graph-file static sites need specific source/question reconciliation; no whole-file audit.'),
        ('S03','Runner model-forward, update ordering and exception release','LOCAL_CONDITIONAL_CORE_ANALYZED','Generic update/ENPU/model kwargs/Engram/metadata executor finally branches.','Live ENPU/backend/graph identities remain unbound; local source is not native epoch proof.'),
        ('S04','Worker graph mode, batch/padding and DP dispatch','EXISTING_PROFILE_CONTRACTS_LINKED_CASE_AUDIT_PENDING','Mode and padded batch/DP decisions.','Reconcile fixed-case dispatch and global rank agreement with existing v16/v21 evidence.'),
        ('S05','Capture/warmup/dummy/profile lifecycle','EXISTING_PROFILE_CONTRACTS_LINKED_FULL_SCOPE_PENDING','Worker dummy,capture,warmup,profile paths.','Actual capture bank/output/phase and all relevant mode cases remain.'),
        ('S06','Worker metadata/input state adjacent planes','CROSS_MODULE_SCOPE_RECONCILIATION_PENDING','Source selector includes M04/M21 preparation and metadata builders.','Reuse their resolved finite contracts; determine remaining graph-owned boundaries rather than read entire runner.'),
        ('S07','Speculation/sampling/bookkeeping adjacent planes','CROSS_MODULE_SCOPE_RECONCILIATION_PENDING','Source selector includes M17-M21 proposal/sample/bookkeeping.','Reuse v33-v43 source contracts; actual output/captured generations remain cross-boundary.'),
        ('S08','Persistent KV/cache/weights/allocation/storage adjacent planes','CROSS_MODULE_SCOPE_RECONCILIATION_PENDING','Source selector includes allocation/reshape/KV spec/load ownership.','Map M01/M10/M24 evidence and graph-visible storage ABI; static file association is not graph ownership.'),
        ('S09','Other runner initialization/control/process adjacent planes','CROSS_MODULE_SCOPE_RECONCILIATION_PENDING','Remaining original worker source associations retained as navigation.','Explicit neighbor ownership/case relevance needed; never silently drop original indexed scope.'),
        ('S10','Named SMLA/QLI metadata producer execution placement','DIRECT_PLACEMENT_DERIVED_FOR_529_NAMED_TASKS','321SMLA+208QLI metadata all direct;0 graph for these types.','Not plan-memory identity or absence of opaque/fused/internal producer work.'),
        ('S11','Actual graph/native phase/cohort/body association','64_PHASE_LINKED_16_UNBOUND;FULL_BANK_BODY_PENDING','80graph executions/132000 tasks and exact submission/root relations.','16phases and actual QLI/SMLA/bank/parameter/task relations require full saved data.'),
        ('S12','All inputs/outputs/pool/workspace/bank owners and lifetimes','PARTIAL_SOURCE_PROTOCOL;ALL_OBSERVERS_PENDING','kwargs/nested tensors,weak outputs,global pool and every last reader/reuse.','Actual storage/alias/capture/address generations not established by debug assertion or fixed shape.'),
        ('S13','Cross-stream/rank/iteration generation protocol','CONDITIONAL_GRAPHS_ANALYZED;ACTUAL_EPOCH_BINDING_PENDING','Metadata producers,CPU param mutators,graph replay,last readers and reuse.','Native parameter epochs and all rank/input-ready/last-reader fences remain critical.'),
        ('S14','Resource/cost/critical-path same-denominator bounds','RESOURCE_BOUNDS_INCOMPLETE','Graph/core/CPU waits,metadata,copy,launch and overlap costs.','No measured sync-removal gain,physical peak/MFU or missing-PMU zero filling.'),
        ('S15','Whole-process frontend/EngineCore/queue transport','FULL_PROCESS_SCOPE_PENDING','Worker graph boundary plus upstream/downstream publication.','Worker trace cannot supply unrecorded EngineCore onCPU/producer-ready/transport costs.'),
        ('S16','External/native/config compatibility and literature','PARTIAL_FROZEN_SOURCE;NATIVE_CONTRACT_PENDING','CANN9.2 native update/weak-reference/backend compatibility.','No SDK-wide audit; exact native epoch question only when needed to prove a chosen retiming.'),
    ]
    ev=['M22-capture-replay-input-output-and-update-contract-v44.json','M22-named-metadata-graph-placement-derivation-v44.json','M22-positional-address-check-domain-proof-v44.json']
    branches=[dict(id='M22-'+i,branch=b,state=s,scope=sc,gap=gap,evidence=ev,whole_module_closed=False,existing_static_site_count=0)for i,b,s,sc,gap in definitions]
    bm={r['id']:r for r in branches}
    def family(r):
        o=r['owner']
        if 'acl_graph.py'in r['source']:
            return 'M22-S01'if o in ['ACLGraphWrapper.__init__','ACLGraphWrapper.__call__','ACLGraphWrapper._updatable_graph_replay']else'M22-S02'
        name=o.split('.')[1]if '.'in o else o
        if name in ['_model_forward','_update_full_graph_params_if_needed']:return'M22-S03'
        if any(s in name for s in ['cudagraph_mode','execution_and_padding','_sync_metadata_across_dp']):return'M22-S04'
        if any(s in name for s in ['capture','dummy','warmup','profile_run']):return'M22-S05'
        if any(s in name for s in ['kv_cache','block_table','load_model','weights','lora','reload']):return'M22-S08'
        if any(s in name for s in ['sample','propos','bookkeeping','logprob','draft']):return'M22-S07'
        if any(s in name for s in ['metadata','prepare_input','update_state','input_batch','seq_lens','positions']):return'M22-S06'
        return'M22-S09'
    notes={
        108:'NONE constructor prohibited; non-graph fallback belongs outer dispatch.',115:'Debug runnable repr only; does not validate runtime banks.',119:'Default options object created only if absent; caller overrides remain distinct.',
        161:'NONE or different runtime mode calls callable directly, including nested wrappers/warmup.',170:'Descriptor-keyed entry allocation; no graph yet.',176:'Capture only if entry lacks graph; subsequent calls replay.',177:'Debug capture logging is not runtime work/gain.',186:'Capture address inventory checks only direct positional tensors.',
        191:'GC/empty-cache patches scoped by ExitStack during capture.',209:'Capture joins offloader and wraps/rethrows errors; failed capture does not publish entry graph/output.',219:'Optional weak-ref output before capture exit requires last-piecewise-output lifetime condition.',228:'Old-stack-signature branch wraps RuntimeError; no actual failure or upgrade request is inferred.',235:'Stream-resource-signature branch wraps RuntimeError; no actual failure inferred.',246:'Super-kernel optimization separate config; not implied by static-kernel flag.',275:'Debug-only positional address check.',277:'Replay inventory omits kwargs/nested containers.',278:'Equality covers listed positional pointers only; runner kwargs call can yield empty lists.',293:'Draft EAGLE flag conjunct, not generic draft detection.',294:'FULL and non-draft-EAGLE determine sync eligibility.',295:'Non-ENPU eligible replay uses host current-stream synchronization; do not remove without full generation/CPU mutation proof.',297:'FULL and eligible backend use updatable path; others direct graph replay.',
        309:'Updatable path requires update_stream.',310:'Draft SharedSource differs from target ContextSource metadata owner.',314:'ENPU orders update→replay, non-ENPU replay→update; native epoch semantics remain opaque.',
        3152:'Generic update requires FULL&&!capturing&&!sparse&&!compress.',3153:'All four conditions retained; sparse kernel name is not live Python flag.',3157:'ENPU branch of generic update includes synchronization.',3201:'Model object must exist.',3203:'Forward context must exist.',3204:'FULL path may bind backend to wrapper.',3205:'Bind backend only when setter supported; actual class remains separate.',3217:'Engram producer only if exposed by model.',3218:'Any capture condition selects graph inputs instead of live per-iteration routing.',3219:'Capture-active/forward-capturing/NPU-capturing are alternative signals; do not infer all from one flag.',3244:'Finally releases in-flight metadata even on forward error.',3245:'Runner ENPU controls generic update before vs after model.',3257:'Release only if executor exists and has an in-flight submission; preserve exception recovery.'}
    reconciled=[]
    for s in sites:
        f=family(s);s['semantic_family']=f;bm[f]['existing_static_site_count']+=1
        if f in ['M22-S01','M22-S03']:
            s.update(review_status='EXACT_GUARD_LOCAL_DATAFLOW_RECONCILED',runtime_binding='CONDITIONAL_SOURCE_DOMAINS;NOT_ALL_LIVE_FLAGS_OR_BANKS',semantic_evidence=ev)
            reconciled.append(dict(id=s['id'],owner=s['owner'],line=s['line'],kind=s['kind'],condition=s['condition'],semantics=notes[s['line']],evidence=ev))
        else:s.update(audit_state='NAVIGATION_ONLY_NOT_AUTOMATIC_SEMANTIC_PROOF',linked_evidence=ev)
    assert len(reconciled)==41
    assert sum(r['existing_static_site_count']for r in branches)==1101 and bm['M22-S02']['existing_static_site_count']==28
    write(O/'M22-original-scope-and-guard-reconciliation-v44.json',dict(
        generated_utc=stamp,original_surface={k:v for k,v in orig.items()if k not in ['indexed_unit_ids','indexed_branch_ids']},
        indexed_units=len(orig['indexed_unit_ids']),indexed_static_sites=1101,source_hashes_verified=hashes,
        guard_dataflow_reconciled=reconciled,local_reconciled_sites=41,
        other_static_sites_not_automatically_proved=1060,source_selector_identity_is_module_proof=False,
        note='Most worker source sites are adjacent-module associations. Their ownership/case evidence remains to be reconciled, not silently excluded or all declared graph branches.',
        whole_module_closed=False,overall_complete=False))
    write(O/'M22-semantic-branch-inventory-v44.json',dict(generated_utc=stamp,branches=branches,static_branch_sites=sites,
        static_site_count=1101,semantic_families=16,automatic_semantic_completion_forbidden=True,
        old_matrix_rows_are_progress_not_semantic_branches=True,whole_module_closed=False,overall_complete=False))

    variants=[
        dict(id='F31-v44-GRAPH-PRODUCER-PLACEMENT',parent='F31',transformation='Compare direct named-plan producers with graph-contained producers for compatible fixed bank/shape/frontiers; current named metadata529DIRECT is baseline evidence, not a claim all preparation outside graph.',
             dependencies=['Opaque plan format/capture legality,actual consumer banks and all input-ready/last-reader/reuse epochs.']),
        dict(id='F31-v44-GENERATION-SYNC-REFINEMENT',parent='F31',transformation='Replace broad synchronization only when complete generation obligations can be met by specific last-reader/ready fences, or independent graph+bank generations; preserve any CPU parameter-mutator host completion requirement.',
             dependencies=['Actual ENPU/backend/update tasks and native update epoch semantics; next iteration data dependencies cannot be speculated away.']),
        dict(id='F31-v44-MULTIBANK-GENERATION-PIPELINE',parent='F31',transformation='Pipeline independent next-generation metadata preparation into a distinct bank/graph instance when inputs are already available, retiring each bank only after every captured/direct consumer.',
             dependencies=['Bank-to-graph pointer binding,finite pool/memory/capture cost,all rank agreement and correct feedback input readiness.']),
    ]
    conditions=[
        ('PLACEMENT','529named metadata DIRECT and0GRAPH do not establish every metadata write/opaque micro-op placement or plan address.'),
        ('CONTEXT','Preserve main/draft,mode,descriptor,graph-pool and source metadata owners; actual ENPU/backends and breakable alternatives remain separate.'),
        ('CAPTURE_SCALARS','Named plan ops include CPU-derived scalar max lengths/counts/layout fields. Preserve their actual generation or prove valid fixed capture bounds; fresh GPU tensors alone do not update captured scalars.'),
        ('ARGS','Fixed addresses must cover kwargs and nested tensors, not only positional debug list; runtime replay ignores new args/kwargs.'),
        ('REFS','Captured output/workspace weak references and first-capture return lifetime are preserved through all later consumers/replays.'),
        ('NATIVE_EPOCH','Do not transpose update/replay from Python order alone; CANN native parameter epochs require an exact contract if changing them.'),
        ('CPU_MUTATORS','A device-stream wait does not block CPU descriptor mutation. Keep host completion or separate immutable native parameter generations when required.'),
        ('READY','Every producer waits its true data dependencies and publishes readiness for the bank generation being read.'),
        ('LAST_READ','Every CPU/GPU/captured/other-stream consumer retires before bank mutation/reallocation/replay overwrite.'),
        ('FEEDBACK','Next-step sequence lengths/accepted history/sampled IDs/CP split cannot be computed before their real dependency; no speculative stale plan reuse.'),
        ('FRONTIERS','Preserve stable metadata task stage/group frontier/input-ready/reuse/release, including zero-query and exception paths.'),
        ('OFFLOAD','Capture preserves sync_prev_onload and joined copy streams; graph scheduling cannot assume offloader absent from source silence.'),
        ('MEMORY','Price number of descriptor×bank graph instances,graph-pool retention and output/workspace lifetime; multibank is not free memory.'),
        ('RANKS','All rank collective participation and globally consistent descriptor/bank decisions remain;16unbound phase graphs stay unassigned.'),
        ('EXCEPTIONS','Capture errors and metadata in-flight releases remain correct; no pending submission leak on failed forward.'),
        ('OVERLAP','Producer placement,sync refinement and multibank pipeline share dependencies/alternatives; do not add their claimed benefits.'),
        ('COST','No current sync latency/CPU exposure/performance gain or hardware peak is inferred from counts,weakref/source calls or missingPMU.')]
    for v in variants:v.update(implemented=False,native_validation=False,critical_path_gain_us=None)
    write(O/'M22-graph-generation-conditional-variants-v44.json',dict(generated_utc=stamp,variants=variants,
        conditions=[dict(id='M22-v44-'+k,requirement=v,actual_current_legality_proved=False)for k,v in conditions],
        new_model_candidates=[],whole_module_closed=False,overall_complete=False))
    g=graph(['generation_inputs_ready','bank_prior_readers_complete','GPU_tensor_producers','CPU_native_parameter_mutators','bank_generation_ready','graph_replay','all_output_and_plan_readers','generation_retired','bank_reuse'],[
        ('generation_inputs_ready','GPU_tensor_producers','NO_STALE_FEEDBACK'),('bank_prior_readers_complete','GPU_tensor_producers','NO_WAR'),
        ('bank_prior_readers_complete','CPU_native_parameter_mutators','HOST_COMPLETION_OR_DISTINCT_NATIVE_OBJECT'),
        ('GPU_tensor_producers','bank_generation_ready','DEVICE_READY_EVENT'),('CPU_native_parameter_mutators','bank_generation_ready','NATIVE_EPOCH_READY'),
        ('bank_generation_ready','graph_replay','MATCH_CAPTURED_BANK_ADDRESSES'),('graph_replay','all_output_and_plan_readers','OUTPUT_WEAKREF_LIFETIME'),
        ('all_output_and_plan_readers','generation_retired','ALL_LAST_READERS'),('generation_retired','bank_reuse','REUSE_FENCE')])
    write(O/'M22-graph-generation-protocol-v44.json',dict(generated_utc=stamp,conditional_generation_graph=g,graphs_acyclic=True,
        native_parameter_epoch_latch_not_assumed=True,implementation=False,whole_module_closed=False,overall_complete=False))
    gates=[]
    for name,old in orig['mandatory_gates'].items():
        gates.append(dict(requirement=name,original_status=old['status'],result='INCOMPLETE_WORK'if name=='resource_performance_and_bounds'else'PARTIAL_PROOF',
            evidence=ev+['M22-original-scope-and-guard-reconciliation-v44.json','M22-semantic-branch-inventory-v44.json','M22-graph-generation-conditional-variants-v44.json'],
            completion_proved=False,note='41local guard dataflows and named producer placement are not original whole-module/all-reader/native epoch/resource/process closure. Conditional opportunities need not be implemented to be analyzed.'))
    write(O/'M22-original-requirement-completion-audit-v44.json',dict(generated_utc=stamp,requirements=gates,result='NOT_COMPLETE',
        source_files=2,indexed_units=142,indexed_static_sites=1101,semantic_families=16,locally_reconciled_guard_sites=41,
        source_selector_not_module_identity=True,whole_module_closed=False,overall_complete=False))
    print(json.dumps(dict(named_metadata_DIRECT=529,named_metadata_GRAPH=0,original_static_sites=1101,semantic_families=16,
        exact_guard_dataflows=41,remaining_graphfile_helper_sites=28,conditional_variants=3,conditions=len(conditions),
        original_gates=8,whole_module_closed=False,overall_complete=False)))


if __name__=='__main__':main()
