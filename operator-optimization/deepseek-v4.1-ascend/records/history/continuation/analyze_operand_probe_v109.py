"""Interpret the authorized own-graph probe without attributing it to saved model tasks."""
from pathlib import Path
from collections import Counter
import hashlib,json,re
from finalize_origin_coverage_v9 import N,O,read,write

def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def main():
    assert not(O/'REVIEW-CHECKPOINT-v109.json').exists()
    first=read(N/'operand-probe166-v109.json');actual=read(N/'operand-probe166-v109b.json')['result']
    result=read(N/'operand-probe-results-v109b/probe-result.json');dump=read(N/'operand-probe-results-v109b/own-tiny-graph.json')
    ec=read(O/'operand-probe-error-code-index-v109b.json');text=Path(ec['local_path']).read_text()
    assert '#define ACL_ERROR_RT_PARAM_INVALID 107000'in text
    assert result['status']=='CAPTURED_METADATA_QUERIED_NO_REPLAY'and actual['attempt']['returncode']==0
    assert 'Read-only file system'in first['result']['attempt']['stderr']
    assert actual['container_guard']['stdout']=='running 2032500\n'
    assert read(O/'operand-probe-error-code-index-v109.json')['container_after']['stdout']=='running 2032500\n'
    assert all('No running processes found in NPU '+str(i)in actual['resource_after']['stdout']for i in range(8))
    tasks=result['tasks'];byseq={t['seq_id']:t for t in tasks};assert len(byseq)==len(dump)==4
    rows=[]
    for d in dump:
        task=byseq[d['args']['Task Id']]
        rows.append(dict(own_sequence_id=task['seq_id'],dump_task_name=d['name'],dump_task_type=d['args']['Task Type'],
            generic_task_type=task['type'],get_params_returncode=task['get_params_returncode'],
            returncode_name='ACL_ERROR_RT_PARAM_INVALID'if task['get_params_returncode']==107000 else'ACL_SUCCESS',
            argsSize=task.get('argsSize'),opInfoSize=task.get('opInfoSize'),args_host_accessible=bool(task.get('isHostArgs'))if task.get('isHostArgs')is not None else None,
            typed_destination_or_write_extent_returned=False))
    kernel=next(t for t in tasks if t['type']==1);rendered=next(d for d in dump if d['args']['Task Id']==kernel['seq_id'])
    words=[int(x,16)for x in re.findall(r'0x[0-9a-fA-F]+',rendered['args']['Kernel Args'])]
    assert len(words)==8 and words.count(result['bool_output_address'])==2 and words.count(24)==3
    assert result['bool_input_address']not in words and kernel['isHostArgs']==0
    env=json.loads(actual['SDK_preflight']['stdout'])
    write(O/'operand-probe-observations-v109.json',dict(status='MEASURED_OWN_CAPTURE_METADATA_ONLY',
        authorization='User:允许最小探针;one model-free metadata probe on an idle device',
        first_process_attempt=dict(status='STOPPED_BEFORE_LIBRARY_LOAD_OR_TORCH_IMPORT',error='Read-only /data/tiankuan/wio output mount',
            elapsed_seconds=first['result']['elapsed_seconds'],NPU_capture_attempt=False),
        recovery='Only output-path restriction changed to own writable /tmp/wio-operand-probe-v109-*;original prepared script/checkpoints immutable.',
        measured=dict(selected_device=0,captured_graphs=1,captured_tasks=4,streams=1,rows=rows,probe_process_seconds=actual['elapsed_seconds'],
            graph_replays=0,model_loads=0,NPU_tensor_payload_copied_to_host=0,opaque_args_dereferenced_by_probe=False,
            SDK_header_sha256=env['header_sha256'],SDK_library_disk_path=env['library_path'],SDK_library_disk_sha256=env['library_sha256']),
        bool_kernel_packet=dict(known_submission='aten::zeros_like(BOOL[24])',kernel_name=rendered['name'],
            argsSize=kernel['argsSize'],opInfoSize=kernel['opInfoSize'],isHostArgs=kernel['isHostArgs'],numBlocks=kernel['numBlocks'],
            rendered_uint64_words=8,known_output_address_occurrences=2,known_input_address_occurrences=0,word_value_24_occurrences=3,
            no_semantic_field_ABI_inferred_from_repetition_or_equal_shape=True),
        controlled_fill_route=dict(submitted_APIs=['aclrtMemsetAsync(count=64B)','aclrtMemsetD32Async(N=16 uint32)'],
            ordered_dump_task_types=['MEMCPY_ASYNC','MEMCPY_ASYNC'],
            exact_saved_MemSet_operator_equivalence_attested=False),
        cleanup=dict(NPU_process_list_empty_on_all_8_cards_after_exit=True,container_PID_before_after=2032500,
            selected_container_not_started_restarted_or_stopped=True,SSH_control_socket_present=(N/'ssh166-probe-v109.sock').exists()),
        debug_dump_ts_dur_used_as_execution_times=False,new_saved_model_native_attributions=0,overall_complete=False))
    write(O/'operand-probe-capability-gates-v109.json',dict(
        resolved=['The installed generic getters and own NPUGraph dump work for this controlled captured graph.',
            'Kernel GetParams succeeds for this BOOL24 ZerosLike task and reports argsSize64,opInfoSize136,isHostArgs0,numBlocks1.',
            'Both controlled API fill submissions appear as MEMCPY_ASYNC in this graph;their GetParams calls return107000 with no ranges.'],
        not_resolved=['GetParams107000 names invalid parameters in the exact SDK header;it does not identify the root cause or prove universal unsupportedness for all DEFAULT or saved MemSet tasks.',
            'The controlled fill route differs from the saved ConstantPadNd-owned MemSet operator;do not transfer native-task equivalence.',
            'A BOOL24 packet with repeated24 and repeated output address does not identify its high-level semantic field or all readers.',
            'argsSize64/opInfoSize136 are not tensor fill length;device args are opaque and were not dereferenced.',
            'No historical process/call/graph/replay/update/storage epoch or MC2 receive counts were recovered.',
            'Debug dump coordinates from a never-replayed graph are not measured device execution duration or latency.'],
        next_evidence=['Exact native operator/provider ABI plus same-call range/field correlation is still needed;do not rerun an equivalent tiny probe for these questions.',
            'A full-model diagnostic would load/replay/infer outside this minimal authorization and requires its own concrete finite plan and authorization.',
            'Continue existing-profile and pinpoint operator-source analysis in parallel with preparing any genuinely necessary missing evidence.'],
        authorization_status='GRANTED_AND_CONSUMED_FOR_THIS_MINIMAL_PROBE;NO_PENDING_REPLY',
        retained_model_gaps=dict(MemSet_regimes=7,MemSet_executions=1272,BOOL24_regimes=2,BOOL24_executions=80,
            MC2_raw_C65536_target_records=3200,MC2_other_unknown_capacities=1272),
        numeric_physical_ceiling_MFU_speedup_or_whole_closure=False,overall_complete=False))
    paths=[N/'probe-occupancy166-v109-before.json',N/'operand-probe-submission-v109.json',N/'operand-probe166-v109.json',
        N/'operand-probe-output-path-check-v109.json',N/'operand-probe-output-recovery-v109.json',N/'operand-probe-submission-v109b.json',N/'operand-probe166-v109b.json',
        N/'operand-probe-error-code-source-v109.json',N/'operand-probe-error-code-source-v109b.json',
        N/'diagnostic-operand-probe-v107/inspect_graph_params.py',N/'diagnostic-operand-probe-v109/inspect_graph_params.py',
        N/'run_operand_probe166_v109.py',N/'run_operand_probe166_v109b.py',
        N/'operand-probe-results-v109b/probe-result.json',N/'operand-probe-results-v109b/own-tiny-graph.json',Path(ec['local_path']),
        O/'operand-probe-result-index-v109.json',O/'operand-probe-result-index-v109b.json',O/'operand-probe-error-code-index-v109.json',O/'operand-probe-error-code-index-v109b.json']
    write(O/'operand-probe-inputs-v109.json',dict(inputs=[pin(p)for p in paths],new_device_work=True,
        new_model_executions=0,at_most_one_NPU_capture_attempt=True,overall_complete=False))
    print(json.dumps(dict(status='INTERPRETED_OWN_CAPTURE',tasks=4,kernel_params_success=1,GetParams_107000=3,
        model_operand_gaps_closed=0,model_graph_replays=0)))

if __name__=='__main__':main()
