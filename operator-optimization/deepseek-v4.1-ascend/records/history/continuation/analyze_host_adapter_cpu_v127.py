"""Reconcile Linux CPU-only adapter tests; never invoke a CANN runtime."""
from pathlib import Path
import hashlib,json
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not (O/'REVIEW-CHECKPOINT-v127.json').exists()
    d=N/'eager-host-adapter-v127'
    c=read(O/'host-adapter-CPU-results-v127c.json')
    b=read(O/'host-adapter-CPU-results-v127b.json')
    a=read(O/'host-adapter-CPU-results-v127.json')
    assert a['status']=='CPU_BUILD_ERROR'
    assert c['status']=='CPU_ADAPTER_TESTS_FINISHED' and len(c['cases'])==23
    rows=[]
    for r in c['cases']:
        hosts=[x for x in r['records']if x['type']=='host_args']
        footers=[x for x in r['records']if x['type']=='summary']
        rows.append(dict(case=r['name'],returncode=r['returncode'],backend=r['stdout'],
            host_records=len(hosts),footer_count=len(footers),footer=footers[0]if footers else None,
            structurally_finalized_log=bool(footers),
            complete_original_operand_capture_claim_allowed=False,
            actual_CANN_or_NPU_test=False))
    write(O/'host-adapter-test-audit-v127.json',dict(
        status='LINUX_AARCH64_CPU_FAKE_RUNTIME_TESTS_RECONCILED',cases=rows,
        builds=len(c['builds']),all_builds_successful=all(x['returncode']==0 for x in c['builds']),
        sanitizer_cases=['asan_one','asan_many','asan_recursive','asan_oversize'],
        forwarding_contract_tested='Original blocks/stream/cfg/packet/placeholder values, backend mutation visible to caller, rc and C errno; fake backend only.',
        prelaunch_snapshot_tested='The observer packet has zero pointer slot before fake relocation; caller sees the fake relocated pointer afterwards. No embedded pointer is dereferenced.',
        quota_test=dict(threads=8,calls_per_thread=20,backend_calls=160,reservations=7,written=7,quota_skips=153),
        byte_budget_test=dict(max_bytes=1024,host_records=0,byte_skips=1,footer_retained=True),
        recursion_test=dict(backend_calls=2,observer_seen=2,host_records=1,reentrant_skips=1,
            last_task_is_nested_not_outer=True,task_identity_requires_independent_same_run_join=True),
        noncoverage_witnesses=[
            'Explicit dlsym on the backend handle calls the fake backend successfully with no adapter file.',
            'Attempt b compiler-local recursion calls the backend twice, but observer seen=1 and reentrant_skips=0. A zero skip counter does not prove one runtime task.',
            'Fork child forwarding succeeds while its trace is intentionally unsupported.',
            'Early _exit leaves a host record without footer; classify incomplete, never accept as complete capture.',
            'A missing output directory still forwards successfully but produces no file.'],
        current_source_attempt='c',real_CANN_runtime_loaded=False,new_device_work=False,
        original_pending_operands_recovered=0,overall_complete=False))
    readiness=dict(
        status='LINUX_CPU_FAKE_RUNTIME_ADAPTER_VALIDATED;REAL_CANN_NOT_TESTED',
        implemented=['Default disabled; explicit ENABLE, absolute output directory and restricted run ID required.',
            'Pre-forward bounded host packet and placeholder copy; no embedded pointer dereference.',
            'Original API pointers, rc and C errno forwarded in CPU fixtures; no model/framework source changed.',
            'RTLD_NEXT plus explicit already-loaded RTLD_NOLOAD backend fallback; no new runtime library loading.',
            'MemSet name filter, per-process owner, thread ID, sequence reservation and optional task/stream getters.',
            'Mutex/TLS guards; total records/bytes/time limits; optional arm file; terminal summary footer.',
            'Resolver-only handshake exposes missing backend without attempting a launch.'],
        validated=dict(platform=c['platform'],CPU_cases=23,compiler_builds=6,ASan_UBSan_cases=4,
            installed_CANN_library_inspection='Pinned SHA and readelf exports only; never dlopen/call this library in tests.',
            forwarded_backend='CPU fake runtime in an independent /tmp directory in the existing container'),
        default_limits=dict(bytes_per_packet=4096,placeholders_per_packet=32,records=56,total_bytes=1048576,seconds=60),
        hard_configuration_limits=dict(records=256,total_bytes=16777216,seconds=3600),
        remaining_gates=[
            'Actual model process public-symbol interposition/provider/library namespace and bypass census; synthetic success does not prove real hooks fire.',
            'Actual CANN forwarding and metadata getter semantics, including internal runtime error state; preserving C errno alone is insufficient.',
            'Same-run full process/rank/device/thread/stream/task/connection/capture/update identity. rc0 proves submission only, not completion or exactly one task.',
            'Actual live host packet/placeholder bytes and provider-specific parameter layout. No universal fixed prefix, pointer slot or byte count assumptions.',
            'Destination allocation/alias generation, storage/workspace/payload roles, actual write/read spans and all-reader retirement.',
            'Captured BOOL24 creation/value/storage/update/readers for 80 graph tasks; this eager MemSet collector cannot close them.',
            'Fresh device-window authorization, idle-device check and owned-process-only minimal smoke procedure before any actual CANN/NPU execution.',
            'Actual application exit/IO loss/selection budget and full observation coverage: footer required; quota/time/name failures and missing files preserve unknown scope.'],
        full_model_capture_ready=False,installed_into_model=False,actual_CANN_interception_tested=False,
        original_eager_MemSet_tasks=1272,original_captured_BOOL_tasks=80,
        original_pending_operands_recovered=0,additional_device_probe_or_model_capture_authorized=False,
        Pad_authorization='GRANTED_AND_CONSUMED',new_device_work=False,overall_complete=False)
    write(O/'host-adapter-collector-readiness-v127.json',readiness)
    write(O/'host-adapter-validation-history-v127.json',dict(events=[
        dict(attempt='a',status='RETAINED_CPU_BUILD_ERROR',reason='-Werror rejected ignored write return value; no test executable/CANN/NPU ran.',
            source='host_args_adapter-attempt-a.c',receipt='host-adapter-CPU166-v127.json'),
        dict(attempt='b',status='CPU_CASES_COMPLETED;RECURSION_COVERAGE_GAP_FOUND',reason='23 cases returned successfully, but compiler-local recursive call bypassed the observer; cannot claim TLS reentry guard was tested.',
            source='fake_runtime-attempt-b.c',receipt='host-adapter-CPU166-v127b.json'),
        dict(attempt='c',status='CPU_CASES_COMPLETED;EXPLICIT_PUBLIC_REENTRY_OBSERVED',reason='Forced RTLD_DEFAULT lookup for the reentry fixture. Both normal/sanitized recursive cases show backend2/seen2/skipped1/hostrecord1; last task is nested.',
            source='fake_runtime.c',receipt='host-adapter-CPU166-v127c.json')],
        model_or_target_operator_source_changed=False,remote_services_or_container_restarted=False,
        SSH_session_close_receipt=str(N/'host-adapter-CPU-session-close-v127.json'),
        new_device_work=False,overall_complete=False))
    inputs=[O/'host-launch-postcheck-result-v126p.json',O/'host-launch-original-API-coverage-v126.json',
        O/'host-launch-installed-ABI-contract-v126.json',O/'host-launch-collector-readiness-v126.json',
        N/'host-adapter-CPU-preflight166-v127.json',N/'host-adapter-CPU-session-close-v127.json',
        N/'run_host_adapter_cpu166_v127.py',N/'eager-host-launch-source-v126/host_packet_snapshot.h',
        N/'eager-host-launch-source-v126/host_packet_snapshot.c']
    for suffix in ['v127','v127b','v127c']:
        inputs.extend([N/('host-adapter-CPU166-'+suffix+'.json'),O/('host-adapter-CPU-results-'+suffix+'.json')])
    inputs.extend(sorted(d.glob('*.c')));inputs.append(d/'host_packet_snapshot.h')
    write(O/'host-adapter-inputs-v127.json',dict(inputs=[pin(p)for p in inputs],questions=[
        'Can a bounded default-disabled adapter preserve live host packet bytes before forwarding without dereferencing device addresses?',
        'Do CPU forwarding/error/budget/thread/loader/reentry fixtures pass, and which success/no-log cases still fail to prove complete coverage?',
        'Which real-runtime identity/layout/lifetime and captured-BOOL gaps remain before fusion legality or native work extent can be concluded?'],
        scope='Diagnostic adapter and CPU fake-runtime only; no framework survey, model/NPU execution or optimization implementation.',overall_complete=False))
    print(json.dumps(dict(status='CPU_RESULTS_RECONCILED',cases=23,new_device_work=False,original_pending_promotions=0)))


if __name__=='__main__':main()
