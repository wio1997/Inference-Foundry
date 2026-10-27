"""Exact annotation→raw task audit. Raw graph visualization times are unused."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def join_marker(marker, nodes):
    q=marker['model_ri_query']
    annotation=json.loads(q['annotation_json'])
    assert annotation=={'q27_marker_id':marker['marker_id'],'q27_marker_schema':1}
    assert q['marker_id']==marker['marker_id']
    assert q['capture_id']==marker['capture_id']
    assert q['graph_python_id']==marker['graph_python_id']
    assert q['device_id']==marker['device']
    assert q['query_phase']=='post_capture_before_graph_reset'
    assert q['capture_status']==1 and q['model_ri_nonnull'] is True
    assert marker['callback_capture_status']==1 and marker['callback_nonnull'] is True
    for k in ['extend_info_rc','task_after_rc','model_ri_get_id_rc',
              'get_streams_rc','capture_rc','callback_rc']:
        assert q[k]==0,k
    assert all(v==0 for v in marker['callback_return_codes'].values())
    assert q['annotation_task_id_unchanged'] is True
    assert q['task_annotation_status']=='annotated_exact_last_task_candidate'
    assert q['task_id_before_annotation']==q['task_id_after_annotation']==marker['task_id']
    assert q['physical_stream_id']==marker['acl_stream_id']
    streams=q['model_streams']
    assert streams and all(s['stream_get_id_rc']==0 for s in streams)
    stream_ids={int(s['model_stream_id']) for s in streams}
    matches=[]
    for i,node in enumerate(nodes):
        args=node.get('args',{})
        ext=args.get('ExtendInfo')
        if isinstance(ext,str):
            try:ext=json.loads(ext)
            except (ValueError,TypeError):continue
        if isinstance(ext,dict) and ext.get('q27_marker_id')==marker['marker_id']:
            matches.append((i,node,ext))
    assert len(matches)==1,('raw_annotation_cardinality',len(matches))
    i,node,ext=matches[0];args=node['args']
    assert ext==annotation
    assert args['Task Type']=='EVENT_RECORD'
    assert node['pid']==str(marker['pid'])+' aclGraph'
    assert int(args['Model Id'])==q['model_ri_id']
    assert int(args['Task Id'])==marker['task_id']
    assert int(args['Stream Id']) in stream_ids
    return {'rank_id':marker['rank_id'],'producer_pid':marker['pid'],
            'capture_id':marker['capture_id'],'graph_python_id':marker['graph_python_id'],
            'device':marker['device'],'marker_id':marker['marker_id'],
            'layer_idx':marker['layer_idx'],'marker_side':marker['marker_side'],
            'raw_node_index':i,'raw_task_key':[int(args[k]) for k in ['Model Id','Stream Id','Task Id']],
            'physical_stream_id':q['physical_stream_id'],
            'RI_stream_ids':sorted(stream_ids),'RI_bridge_status':q['ri_bridge_status'],
            'join_method':'unique_exact_ExtendInfo_JSON_marker; no physical/model stream inference'}

def audit(snapshot, probe, label):
    snapshot=Path(snapshot)
    d=snapshot/probe
    state=json.loads((d/'state.json').read_text())
    assert state['status']=='raw_collected_actual_graphs_reset_pending_native_audit'
    assert state.get('cleanup') and not state.get('cleanup_error')
    source=json.loads((d/'endpoint-rows.json').read_text())
    anchors=[z for z in source if z['event']=='layer_public_event_marker']
    summaries=[z for z in source if z['event']=='layer_public_event_summary']
    assert len(anchors)==512 and len(summaries)==4
    assert {z['rank_id'] for z in summaries}=={0,1,2,3}
    cleanup={z['rank_id']:z for z in state['graph_cleanup_receipts']}
    assert len(cleanup)==4
    receipts=[]
    for p in (snapshot/'graph-lineage/vllm'/label/'graphs').glob('pid-*/receipts/*.receipt.json'):
        receipts.append((p,json.loads(p.read_text())))
    assert len(receipts)==4
    joined=[];inputs=[{'path':str(d/'state.json'),'sha256':sha(d/'state.json')},
                      {'path':str(d/'endpoint-rows.json'),'sha256':sha(d/'endpoint-rows.json')}]
    for rank in range(4):
        rows=[z for z in anchors if z['rank_id']==rank]
        assert len(rows)==128 and len({z['marker_id'] for z in rows})==128
        assert {(z['layer_idx'],z['marker_side']) for z in rows}=={(i,s) for i in range(64) for s in ['pre','post']}
        owners={(z['pid'],z['graph_python_id'],z['capture_id']) for z in rows}
        assert len(owners)==1
        pid,graph,capture=next(iter(owners))
        candidates=[(p,r) for p,r in receipts if r['rank_id']==rank and r['pid']==pid and r['graph_python_id']==graph]
        assert len(candidates)==1
        receipt_path,receipt=candidates[0]
        assert receipt['capture_end_status']=='success' and receipt['dump_status']=='success'
        raw_path=snapshot/Path(receipt['raw_path']).relative_to('/task/full-map-20261025')
        raw=json.loads(raw_path.read_text());nodes=raw if isinstance(raw,list) else raw['traceEvents']
        rank_rows=[join_marker(z,nodes) for z in rows]
        assert len({tuple(z['raw_task_key']) for z in rank_rows})==128
        c=cleanup[rank]
        assert c['graph_python_id']==graph and c['status']=='reset_drained_and_events_released'
        assert c['reset_or_discarded'] is True and c['device_drained_before_and_after_reset'] is True
        assert c['release']['released'] is True
        inputs.extend([{'path':str(raw_path),'sha256':sha(raw_path),'bytes':raw_path.stat().st_size},
                       {'path':str(receipt_path),'sha256':sha(receipt_path)}])
        joined.extend(rank_rows)
    return {'status':'executor_verified_512_exact_raw_EVENT_RECORD_identities',
            'task_id':'Q27-FULL-MAP-20261025','generation':3,'executor_audit':True,
            'by_rank':dict(Counter(z['rank_id'] for z in joined)),
            'rows':joined,'inputs':inputs,
            'scope':'Marked TP4 model short diagnostic; raw graph identity only. No raw graph ts/dur used.',
            'limitations':['Native trace/replay matching remains a separate gate.',
                           'No exclusive-kernel or all-stream critical-path claim.',
                           'No uninstrumented performance or greedy/quality equivalence claim.']}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--snapshot',required=True)
    a.add_argument('--probe',default='vllm-task-tag-model-gen81-exec-g3')
    a.add_argument('--label',default='lineage-vllm-fx-model-g81-exec-g3')
    a.add_argument('--output',required=True);args=a.parse_args()
    result=audit(args.snapshot,args.probe,args.label)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','inputs']}))
