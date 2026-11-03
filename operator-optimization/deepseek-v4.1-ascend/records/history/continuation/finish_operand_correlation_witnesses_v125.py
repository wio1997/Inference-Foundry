"""Concrete nonidentity witnesses from already-read profiler and own-Pad metadata."""
from collections import Counter
from pathlib import Path
import hashlib,json
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'REVIEW-CHECKPOINT-v125.json').exists()
    native=read(O/'operand-correlation-native-records-v125.json')['records']
    cpu=read(O/'operand-correlation-CPU-stacks-v125.json')
    root=next(c for c in cpu['BOOL24_zeros_like_calls']if c['database_index']==0)
    linked=root['exact_native_links'];assert len(linked)==1 and not linked[0]['is_one_of_original_pending_tasks']
    pending=next(r for r in native if r['database_index']==0 and r['operator']=='ZerosLike')
    assert linked[0]['TASK']['globalTaskId']!=pending['TASK']['globalTaskId']
    dump=read(N/'native-pad-probe-results-v120b/own-native-pad-graph.json')
    probe=next(e for e in dump if e['name'].startswith('MemSet_'))
    colliding=[r for r in native if r['operator']=='ZerosLike'and r['TASK']['modelId']==probe['args']['Model Id']]
    assert colliding and all(str(r['TASK']['globalPid'])!=probe['pid'].split()[0]for r in colliding)
    witness=dict(inputs=[pin(p)for p in [O/'operand-correlation-native-records-v125.json',O/'operand-correlation-CPU-stacks-v125.json',N/'native-pad-probe-results-v120b/own-native-pad-graph.json']],
        same_shape_false_field_match=dict(database_index=0,CPU_zeros_like=root['root'],CPU_stack=cpu['stacks'][root['root']['stack_ref']],
            exact_linked_eager_task=linked[0],unresolved_graph_task=pending,
            conclusion='Same BOOL24 zeros_like shape/name does not identify the prefix graph field. Exact CPU backend group maps to a separate eager native task.'),
        cross_process_model_id_collision=dict(own_probe_model_id=probe['args']['Model Id'],own_probe_pid=probe['pid'],
            own_probe_stream=probe['args']['Stream Id'],own_probe_task=probe['args']['Task Id'],
            original_pending_tasks_with_same_numeric_model_id=len(colliding),original_process_ids=sorted({r['TASK']['globalPid']for r in colliding}),
            conclusion='Displayed modelId is not globally unique. Use host/run/process/rank/device/capture/update epoch plus stream/task identity;do not join the earlier saved model to the later own probe by numeric ID.'),
        CPU_BOOL24_leaf_stack_counts=dict(Counter(cpu['stacks'][c['root']['stack_ref']][0]['frame']for c in cpu['BOOL24_zeros_like_calls'])),
        original_pending_reconciliations=0,new_device_work=False,overall_complete=False)
    write(O/'operand-correlation-identity-witnesses-v125.json',witness)
    print(json.dumps(dict(status='WITNESSES_RECORDED',same_shape_tasks_distinct=True,numeric_model_id=probe['args']['Model Id'],
        cross_process_collisions=len(colliding),new_device_work=False)))


if __name__=='__main__':main()
