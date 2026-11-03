"""Record the tested CPU core and explicit remaining native-adapter gates."""
from pathlib import Path
import hashlib,json,subprocess
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'REVIEW-CHECKPOINT-v126.json').exists()
    d=N/'eager-host-launch-source-v126';cpu=read(O/'host-packet-CPU-core-result-v126.json')
    assert cpu['status']=='PASS'and cpu['ASan_UBSan']=='PASS'
    api=read(O/'host-launch-original-API-coverage-v126.json')
    assert api['nested_host_args_event_count_distribution']=={'1':1272}
    docs=read(d/'http-index.json');assert all(r['status']==200 for r in docs['sources'])
    inputs=[O/'operand-correlation-native-records-v125.json',O/'operand-correlation-consequences-v125.json',
        O/'operand-correlation-postcheck-result-v125p.json',O/'host-launch-original-API-coverage-v126.json',
        O/'host-launch-installed-ABI-contract-v126.json',O/'host-packet-CPU-core-result-v126.json',
        O/'native-pad-packet-and-store-contract-v120.json',N/'native-pad-probe-results-v120b/own-native-pad-graph.json',
        N/'decode_memset_metadata_v111.py',d/'http-index.json',d/'host_packet_snapshot.h',d/'host_packet_snapshot.c',d/'selftest_snapshot.c']
    inputs += [Path(r['path'])for r in docs['sources']]+[Path(r['path'])for r in cpu['build_artifacts']]
    write(O/'host-launch-inputs-v126.json',dict(inputs=[pin(p)for p in inputs],
        questions=['Do original eager MemSet launches actually traverse a host-argument API?',
            'Can a bounded CPU copy/explicit-placeholder core avoid dereferencing device pointers?',
            'Which live loader/provider/correlation/field/lifetime gates still prevent claiming original operands are recovered?'],
        no_remote_NPU_model_or_CANN_library_execution=True,overall_complete=False))
    dependencies=subprocess.check_output(['/usr/bin/otool','-L',str(d/'cpu-build/snapshot.dylib')],text=True)
    assert 'Ascend'not in dependencies and 'torch'not in dependencies
    write(O/'host-launch-collector-readiness-v126.json',dict(
        status='CPU_COPY_CORE_VALIDATED;NATIVE_ADAPTER_NOT_INSTALLED_OR_TESTED',
        profile_coverage=dict(eager_MemSet_tasks=1272,exact_generic_launch_links=1272,same_thread_unique_nested_HostArgs_APIs=1272,
            typed_destination_or_write_bytes_in_saved_profile=0),
        implemented=['Disabled/device-source guards before any payload read','4096B packet and32placeholder per-copy bounds',
            'Disjoint owned output and unchanged input buffers','Pointer-slot/data-offset bounds and unique inline-relocation lookup',
            'No embedded-pointer dereference,allocator,logging,loader or CANN call in the core'],
        verified=dict(host_platform=cpu['platform'],ASan_UBSan=cpu['ASan_UBSan'],placeholder_size=cpu['placeholder_ABI_bytes'],
            snapshot_size=cpu['ctypes_snapshot_bytes'],pure_library_dependencies=dependencies,
            fixture='Constructed host packet based on prior own v120 post-relocation bytes;not a newly observed host call.'),
        native_adapter_gates=[
            'Match installed header/library/provider binaries and confirm actual Linux symbol/interposition path;public declarations/profile API names do not prove a future preload hook will fire.',
            'Enter only a genuine live aclrtLaunchKernelWithHostArgs call with valid supplied host spans. Never infer host-readable ownership from GetParams isHostArgs0/device address or a JSON label.',
            'Copy packet and placeholder array before forwarding. Preserve original argument pointers/bytes/cfg/stream and exact API result. Forwarding/loader behavior is not implemented or tested by this pure core.',
            'Bind actual function/provider and parameter layout;no universal slot8/data24 or3-pointer packet-size rule. Unknown or ambiguous records remain raw/unknown.',
            'Record process/rank/device/thread/stream/task/capture/update epochs and same-run API result/correlation. LastTaskId is current-thread/user-stream scoped,can fail or repeat;ambiguity must not become a match.',
            'Add bounded target selection and run/window budget before deployment. Snapshot counts,total output bytes and overall time limits are not enforced by this one-copy primitive.',
            'Recover destination allocation/alias epochs,workspace versus payload role and all overlapping readers/writers. Packet presence and a decoded source span alone do not prove safe zero elision.',
            'The80 captured BOOL tasks still need capture-creation/field/storage/reader evidence;this eager route alone does not resolve them.'
        ],
        proposed_unexecuted_adapter_limits=dict(bytes_per_packet=4096,placeholders_per_packet=32,
            selected_records_per_process=56,total_trace_bytes_per_process=1048576,
            note='Only the first two per-copy bounds are implemented. No new diagnostic window or resource use is authorized by these proposed values.'),
        source_requested_fixture_span_bytes=32,original_pending_operands_recovered=0,
        additional_device_probe_or_model_capture_authorized=False,Pad_authorization='GRANTED_AND_CONSUMED',
        new_device_work=False,implementation_is_diagnostic_only=True,model_or_operator_optimized=False,overall_complete=False))
    print(json.dumps(dict(status='READINESS_RECORDED',observed_host_args_API_coverage=1272,CPU_core='PASS',
        native_adapter='NOT_INSTALLED_NOT_TESTED',original_operands_recovered=0,new_device_work=False)))


if __name__=='__main__':main()
