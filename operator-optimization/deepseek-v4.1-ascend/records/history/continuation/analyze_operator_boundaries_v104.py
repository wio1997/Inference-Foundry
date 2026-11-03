"""Reconcile only implementation selection and operator readiness at M00/M03 boundaries."""
from pathlib import Path
from collections import Counter
import hashlib,json,statistics
from finalize_origin_coverage_v9 import N,O,W,read,write

CACHE=N/'original-profile-cache-v53'
def pin(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def quantile(values,q):return sorted(values)[int((len(values)-1)*q)]
def stat(values):return dict(n=len(values),minimum=min(values),median=statistics.median(values),p90=quantile(values,.9),maximum=max(values))

def main():
    assert not(O/'REVIEW-CHECKPOINT-v104.json').exists()
    matrix=read(O/'module-phase-branch-matrix-v103.json')
    runner=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'
    ls=runner.read_text().splitlines();ranges=[(2229,2242),(2280,2286),(2290,2367)]
    parts=[dict(**pin(runner),segments=[dict(first_line=a,last_line=b,source=ls[a-1:b])for a,b in ranges])]
    cap_path=CACHE/'analysis/profile-capabilities-index-v12-v2.json';cap=read(cap_path)
    installed={e['source_sha256']for f in cap['metadata']for e in f['entries']if e['event']=='installed'}
    assert installed=={pin(runner)['sha256']}
    write(O/'operator-boundary-source-contract-v104.json',dict(parts=parts,
        saved_installed_model_runner_source_hash_matches=True,
        facts=['The normal LM path handles optional KV preemptions and updates persistent batch state before checking local zero scheduled tokens.',
            'At local zero tokens, external_launcher plus DP>1 conditionally executes _dummy_run(1,skip_gdn_state_update=True) to coordinate DP; then returns empty output or invokes kv_connector_no_forward.',
            'The source does not identify the actual executor/KV/EC flags in each recorded call. Zero scheduler tokens alone therefore do not prove dummy execution, absence of device work, no state changes, or no collective obligations.',
            'A later defensive guard checks positive total but empty/zero persistent-batch token list before _prepare_inputs. State/error/transfer/order semantics must be preserved; this is not permission to convert invalid metadata into successful empty output.',
            'Current source SHA match is saved on-disk identity, not a direct proof of every loaded callback/selector object or dynamic predicate.'],
        new_target_source_files=0,new_public_documents=0,no_framework_catalogue_or_new_runtime_work=True,overall_complete=False))

    streams=[];zeros=[];files=[];all_total=Counter();request_counts=Counter();resumed=0
    for descriptor in cap['metadata']:
        p=CACHE/'runtime/host-profile'/Path(descriptor['path']).name
        assert pin(p)['sha256']==descriptor['sha256']
        entries=[json.loads(s)for s in p.read_text().splitlines()];assert entries==descriptor['entries']
        enters=[e for e in entries if e['event']=='execute_enter'];exits={e['call_id']:e for e in entries if e['event']=='execute_exit'}
        key={(e['data_parallel_rank'],e['rank'],e['pid'])for e in enters};assert len(key)==1
        dp,tp,pid=next(iter(key));totals=Counter();reqs=Counter();zero_calls=[]
        for e in enters:
            assert sum(e['scheduled_tokens'].values())==e['total_scheduled_tokens']
            x=exits[e['call_id']];assert x['error_type']is None
            totals[e['total_scheduled_tokens']]+=1;reqs[len(e['scheduled_tokens'])]+=1
            resumed+=bool(e['cached_request_state']['resumed_req_ids'])
            if e['total_scheduled_tokens']==0:
                assert e['scheduled_tokens']=={} and e['cached_request_state']['req_ids']==[] and e['new_request_state']==[]
                r=dict(dp_rank=dp,local_tp_rank=tp,pid=pid,call_id=e['call_id'],label=e['label'],
                    enter_monotonic_ns=e['monotonic_ns'],exit_monotonic_ns=x['monotonic_ns'],
                    wrapper_wall_ns=x['wrapper_wall_ns'],wrapper_thread_cpu_ns=x['wrapper_thread_cpu_ns'],
                    source_branch_actual_executor_KV_EC_predicates_attested=False)
                zeros.append(r);zero_calls.append(e['call_id'])
        streams.append(dict(dp_rank=dp,local_tp_rank=tp,pid=pid,execute_entries=len(enters),zero_scheduled_call_ids=zero_calls,
            scheduled_total_token_histogram={str(k):v for k,v in sorted(totals.items())},
            scheduled_request_count_histogram={str(k):v for k,v in sorted(reqs.items())}))
        all_total.update(totals);request_counts.update(reqs);files.append(pin(p))
    assert len(streams)==16 and len(zeros)==64 and all(len(s['zero_scheduled_call_ids'])==4 for s in streams)
    assert sum(all_total.values())==3712 and resumed==0
    write(O/'operator-boundary-saved-work-domain-v104.json',dict(metadata_files=files,streams=streams,
        zero_scheduled_calls=zeros,execute_entries=3712,zero_scheduled_entries=64,positive_scheduled_entries=3648,
        source_model_runner_sha256=pin(runner)['sha256'],
        scheduled_total_token_histogram={str(k):v for k,v in sorted(all_total.items())},
        scheduled_request_count_histogram={str(k):v for k,v in sorted(request_counts.items())},
        resumed_ids_present_entries=0,zero_call_wrapper_wall_ns=stat([r['wrapper_wall_ns']for r in zeros]),
        zero_call_wrapper_thread_cpu_ns=stat([r['wrapper_thread_cpu_ns']for r in zeros]),
        evidence_grade='MEASURED_SAVED_SCHEDULER_SCALARS_AND_INSTRUMENTED_HOST_WRAPPER',
        missing=['finished_req_ids/free or reused block IDs','actual executor/KV/EC predicates','all-rank same-collective-iteration input bounds',
            'dummy/forward operator execution within these calls','device completion and cause-specific host time'],
        no_native_phase_or_zero_device_work_promotion=True,
        limitation='Counts include TP process replicas and empty calls outside/between capture windows. They are not unique logical steps, device row counts or whole-request work. Wrapper spans include instrumentation and are neither NPU completion times nor removable latency.',overall_complete=False))

    scopes={
        'M00':[
            ('Frozen model/hardware/CANN/quantization/parallel/graph-mode selection and recorded implementation identity.',
             ['M21-captured-config-and-selected-dispatch-contract-v42.json','sampling-load-provenance-v102.json','host-profile-summary-v12.json'],
             'Requested launch fields/source hashes do not attest every loaded binary/selector or alternative runtime predicate. Driver/firmware differences are accepted by the user; preserve CANN and operand-ABI evidence.', ['M04','M22','M23']),
            ('Dense, RMS and quantized projection implementation choices; distinct BF16/FLOAT/INT8 contracts and fallback domains.',
             ['projection-operator-numerical-contract-v61.json','norm-quant-operator-source-contract-v94.json','typed-capacity-primary-contract-v89.json'],
             'Native mode, rounding, applicable fallback and architecture capacity binding remain conditional; same shape/name does not identify arithmetic.', ['M06','M07','M11','M15','M16']),
            ('SWA/CSA, QLI modes, context-parallel class and architecture-specific attention implementation choices.',
             ['attention-indexer-native-numerical-contract-v60.json','M19-current-forward-patterns-and-dispatch-v35.json','index-context-operator-contract-v64.json'],
             'Keep exact class/override and metadata/readiness contracts; actual selected values and loaded native binary are not fully bound.', ['M09','M10','M19']),
            ('Fused MC2 versus alternate MoE/shared-expert protocol, active rows, routing types and capacity-clipping behavior.',
             ['receiver-budget-operator-source-contract-v97.json','MC2-attribute-to-ceiling-contract-v98.json','shared-layout-source-index-v78.json'],
             'Recorded C=65536 binds 3200 target executions, not receiver demand or all-rank same-call geometry;1272 capacities and loaded ABI/placement/publication remain unbound.', ['M12','M13','M23']),
            ('Native/custom HC pre/post/collapse and DSpark decoder/auxiliary implementation selection, including no-query/dummy cases.',
             ['HC-Engram-branch-inventory-v84.json','DSpark-tail-branch-inventory-v85.json','M19-current-forward-patterns-and-dispatch-v35.json'],
             'Configuration and positive native calls support local branches; they do not exhaust runtime selectors, all observers or zero/dummy variants.', ['M06','M15','M18','M19']),
            ('Engram hash/slot/history/UVA format and gate choices across the two Engram layers, retaining parameter generations.',
             ['state-engram-operator-contract-v62.json','HC-Engram-branch-inventory-v84.json','Engram-HC-operator-source-contract-v84.json'],
             'Preserve BF16/INT8/MXFP8 format, ownership, slot/state epochs and rotated/native/fallback distinctions. Active constexpr/row values are not all attested.', ['M14','M24']),
            ('Checkpoint load/quant callback/static expert placement and runtime EPLB parameter-layout publication, with initialization separate from steady execution.',
             ['F21-interface-branch-ledger-v52.json','F21-initial-map-and-attribute-helper-contract-v53.json','F21-pinned-source-edge-availability-v53.json'],
             'Actual callback/loader/slot generation and initial I/O costs remain unknown. Loading determines later operator operands; no startup cost or fixed placement improvement is inferred from decode traces.', ['M12','M23','M24']),
            ('Sampler backend, greedy/mixed/random/entropy/synthetic and raw/logprob implementation guards.',
             ['sampling-operator-guard-contract-v102.json','recovery-fallback-source-and-applicability-v101.json','M17-probability-consumer-and-RNG-contract-v33.json'],
             'Explicit-zero request intent is pinned; effective nonempty all-greedy and all-reader/loaded-body identity remain conditional. Preserve pre-return entropy/RNG/errors.', ['M16','M17'])],
        'M03':[
            ('Scheduled request/token rows, prefix reuse and chunk boundaries as input geometry for actual operators, not padding capacity.',
             ['operator-boundary-saved-work-domain-v104.json','sampling-saved-request-domain-v102.json','H02-CP-reuse-contract-v12.json'],
             '3712 process entries describe scheduled rows, not all processed/active/NPU rows. Prefix/new/cached metadata and padded geometry must stay distinct.', ['M02','M04','M21']),
            ('DP/TP/EP participation, global padding and input/metadata publication readiness before dependent collectives.',
             ['shared-layout-source-index-v78.json','communication-operator-contract-v66.json','M21-slot-coordinate-draft-cache-and-QLI-consumer-contract-v40.json'],
             'Same-call all-rank bounds/group generation/participant readiness are not proved by equal shapes, timestamps or rank ordinals; zero local query work need not cancel global obligations.', ['M04','M21','M23']),
            ('Async response/buffer ownership and FIFO publication as operator-consumer/lifetime constraints.',
             ['CPU-process-coverage-contract-v17.json','M25-output-merge-and-terminal-contract-v53.json','module-phase-branch-matrix-v103.json'],
             'Historical169 response polling is retained as historical evidence. Current166 producer-ready/IPC/CPU costs remain unmeasured; no wait fraction or queue sample becomes fusion gain.', ['M22','M24','M25']),
            ('Actual draft confidence/acceptance lengths and next-ID publication determine later operator demand and state snapshots.',
             ['M20-Markov-confidence-and-verification-contract-v36.json','M21-slot-coordinate-draft-cache-and-QLI-consumer-contract-v40.json','DSpark-tail-reader-closure-v85.json'],
             'Scheduled spec capacity5 is not actual accepted/draft count; CPU/device publication and all downstream confidence/raw/aux readers remain explicit.', ['M16','M17','M18','M20','M21']),
            ('Zero-local-scheduled-token, cache/encoder-only, defensive empty-batch and DP-dummy branches retain their preceding state/transfer effects.',
             ['operator-boundary-source-contract-v104.json','operator-boundary-saved-work-domain-v104.json','M21-CP-target-dummy-and-draft-source-scope-v40.json'],
             '64 saved zero-token entries do not attest no device work or actual dummy execution. Finished/block-free fields and executor/KV/EC predicates are missing; _update_states precedes zero return.', ['M04','M05','M10','M21','M23']),
            ('Request removal/resumption, accepted/rejected rollback, slot reuse and bank writer epochs across iterations.',
             ['M21-slot-state-branch-ledger-v38.json','M21-slot-coordinate-draft-cache-and-QLI-consumer-contract-v40.json','ring-storage-source-contract-v79.json'],
             'No resumed IDs in this saved event surface is not impossibility of future resumption. Empty scheduling does not prove no finished requests, free/reused blocks or outstanding readers.', ['M02','M04','M08','M21','M24']),
            ('Parent/request parameter changes, final output/logprob readers and termination acknowledgements constrain tail pruning and reuse.',
             ['M25-output-merge-and-terminal-contract-v53.json','sampling-case-fusion-and-ceiling-routing-v102.json','M25-output-terminal-dependency-source-v53.json'],
             'Retain selected output rank, raw/logprob/request mutation/termination readers and generation acknowledgements. No broad framework cost audit is implied.', ['M01','M16','M17','M25'])]}
    inventory={};all_evidence=set()
    for mid,rows in scopes.items():
        inventory[mid]=[]
        for i,(branch,evidence,gap,owners)in enumerate(rows,1):
            for f in evidence:assert(O/f).exists(),f
            all_evidence.update(evidence)
            inventory[mid].append(dict(id=f'{mid}-OP-{i:02d}',branch=branch,
                state='OPERATOR_SELECTION_OR_READINESS_SCOPE_RECONCILED;DYNAMIC_EVIDENCE_GAPS_OPEN',
                evidence=evidence,gap=gap,operator_owner_modules=owners,
                scope='Only operator implementation/operand/demand/readiness/state-consumer effects; existing detailed owner-module contracts remain authoritative. Not all runtime branches or framework behavior closed.'))
    write(O/'operator-boundary-semantic-inventory-v104.json',dict(modules=inventory,semantic_rows=15,
        exhaustive_runtime_branch_claim=False,no_new_fusion_or_performance_discovery_from_reformatting=True,overall_complete=False))
    mapping={'M00':[[1,7],[2,3,4,5,6,8],[3,5],[1,2,3,4,5,6,8],[1],[4],[1],[2,4],[6],[3,5],[7],[7],[7],[7]],
             'M03':[[3],[3],[3],[1,5],[1,2,3],[2,3],[4],[4,6],[6],[5,6],[2,6],[7]]}
    cross=[]
    for mid,indices in mapping.items():
        old=next(m for m in matrix['modules']if m['id']==mid)['branches'];assert len(old)==len(indices)
        for j,(r,targets)in enumerate(zip(old,indices)):
            cross.append(dict(module=mid,old_ordinal=j,old_record=r,semantic_rows=[f'{mid}-OP-{x:02d}'for x in targets],
                disposition='RECONCILED_OPERATOR_SCOPE_WITH_HISTORICAL_EVIDENCE_AND_OPEN_GAPS_RETAINED',old_record_closed=False))
    assert len(cross)==26
    write(O/'operator-boundary-history-crosswalk-v104.json',dict(parent='module-phase-branch-matrix-v103.json',crosswalk=cross,
        old_chronological_rows=26,new_operator_semantic_rows=15,prior_total_rows=364,new_total_rows=353,
        semantic_modules_after=26,dropped_old_records=0,scope_was_not_redefined_as_complete=True,overall_complete=False))
    cases=[dict(external_launcher=e,DP_more_than_one=d,has_KV_transfer=k,
        local_scheduled_tokens=0,prebranch_update_states_retained=True,
        source_requires_DP_dummy_before_return=e and d,
        source_return='KV_CONNECTOR_NO_FORWARD'if k else'EMPTY_MODEL_OUTPUT',
        actual_predicates_attested=False)for e in [False,True]for d in [False,True]for k in [False,True]]
    write(O/'operator-boundary-demand-and-theory-contract-v104.json',dict(zero_local_LM_source_cases=cases,
        entry_domain='Normal LM path after state/error/EC early guards; preserve preceding KV preemption handling and persistent-state updates. Source-case table is not actual per-call branch attestation.',
        fusion_contracts=['Global-zero or no-query specialization needs all relevant state/transfer/participant obligations, not only local scheduler zero.',
            'Parameter/selector epochs belong in fused operand/cache validity; initialization work and amortized steady execution use separate phases.',
            'Buffer publication/order constrains consumer eligibility and lifetime; it does not force every producer to finish as a whole when a legal tile interface exists.',
            'Reuse owner-module algorithm/work/precision analyses once. M00/M03 are crosswalk boundaries, not another copy of matrix/communication FLOPs.'],
        theory=dict(status='INCOMPLETE',
            source_zero_local_query_obligation='Local query count0 does not specify all semantically necessary state/memory/transfer/coordination work.',
            measured_comparison='Saved zero-call instrumented wrapper wall/thread CPU spans are available; cause-specific/native completion/primary work and common typed physical capacity are missing.',
            lowest_module_MFU='NOT_ESTABLISHED; absent primary matrix work is not zero whole-module utilization or a valid rank against compute modules.',
            no_numeric_latency_lower_or_speedup_or_free_initialization=True),
        new_native_phase_bindings=0,whole_module_closed=False,overall_complete=False))
    evidence=sorted(all_evidence-{f for f in all_evidence if f.endswith('v104.json')})
    inputs=[O/'module-phase-branch-matrix-v103.json',cap_path,runner]+[O/f for f in evidence]
    write(O/'operator-boundary-inputs-v104.json',dict(inputs=[pin(p)for p in inputs]+files,
        new_source_files_from_target=0,new_public_documents=0,
        new_scope='Reconcile two remaining chronological boundaries into operator selection/readiness crosswalks, with a new saved zero-scheduled-call census and narrowly checked zero-work source obligations.',
        no_server_model_Torch_NPU_capture_service_or_implementation=True,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',zero_scheduled_entries=64,semantic_rows=15,chronological_rows_reconciled=26,total_rows_after=353,semantic_modules=26,overall_complete=False)))

if __name__=='__main__':main()
