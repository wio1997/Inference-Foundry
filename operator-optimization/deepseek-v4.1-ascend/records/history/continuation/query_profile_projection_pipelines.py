"""Query every retained target scope, retaining incomplete extraction cases."""
from pathlib import Path
from decimal import Decimal
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import statistics

R = Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
OUT = Path(__file__).parent / 'profile-first-20261031'
SIDE = R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'


def read(rel):
    return json.loads((R/rel).read_text())


def dims(text):
    return [tuple(int(v.strip()) for v in part.split(',')) if part.strip() else () for part in text.strip().strip('"').split(';')]


def describe(values):
    return {'n':len(values), 'min':min(values), 'median':statistics.median(values), 'max':max(values)} if values else None


def main():
    pipeline = []
    selected_f10 = []
    audit = []
    scopes_total = 0
    rank_count = 0
    for line in SIDE.open():
        obj=json.loads(line)
        if obj.get('kind')!='rank': continue
        rank=obj['data']; rank_count+=1
        audit.append({'ep':rank['ep'], 'rank':rank['rank'], 'status':rank['status'], 'counts':rank['counts']})
        for layer in rank['layers']:
            scopes_total+=1
            base={'ep':layer['ep'], 'layer':layer['layer'], 'scope_id':layer['layer_scope_id'], 'digest':layer['scheduler_digest']}
            tasks=[]
            for group in layer['operator_groups']:
                for t in group['tasks']:
                    c=t['csv']
                    s=int(Decimal(t['device_start_us'])*1000)
                    duration=int(Decimal(t['trace_duration_us'])*1000)
                    tasks.append({'task_id':c['Task ID'], 'flow':t['flow_id'], 'type':c['Type'], 'name':c['Name'],
                                  'stream':c['Stream ID'], 'start_ns':s, 'end_ns':s+duration, 'duration_ns':duration,
                                  'input_shapes':c['Input Shapes'], 'output_shapes':c['Output Shapes'],
                                  'input_dtypes':c['Input Data Types'], 'output_dtypes':c['Output Data Types']})
            tasks.sort(key=lambda t:(t['start_ns'],t['stream'],t['task_id']))
            first=[t for t in tasks if t['type']=='TransposeBatchMatMul']
            second=[t for t in tasks if t['type'] in ['MatMulV2','MatMulV3'] and len(dims(t['input_shapes']))>=2
                    and len(dims(t['input_shapes'])[0])==2 and dims(t['input_shapes'])[0][1]==1024
                    and dims(t['input_shapes'])[1]==(5120,1024)]
            row={**base, 'f16_first_count':len(first),'f16_second_count':len(second)}
            if len(first)==len(second)==1:
                a,b=first[0],second[0]
                row['f16_pair']={'first':a,'second':b, 'same_stream':a['stream']==b['stream'],
                                 'gap_ns':b['start_ns']-a['end_ns'],
                                 'output_input_row_match':dims(a['output_shapes'])[0][0]==dims(b['input_shapes'])[0][0]}
            act=[t for t in tasks if t['type']=='DequantSwigluQuant']
            up=[t for t in tasks if t['type']=='QuantBatchMatmulV3' and dims(t['output_shapes'])[0][-1:]==(4608,)]
            down=[t for t in tasks if t['type']=='QuantBatchMatmulV3' and len(dims(t['input_shapes'])[0])==2 and dims(t['input_shapes'])[0][1]==2304
                  and dims(t['output_shapes'])[0][-1:]==(5120,)]
            row['f22_counts']={'up':len(up),'act':len(act),'down':len(down)}
            if len(up)==len(act)==len(down)==1:
                a,b,c=up[0],act[0],down[0]
                row['f22_chain']={'up':a,'activation':b,'down':c,
                                  'all_same_stream':len({a['stream'],b['stream'],c['stream']})==1,
                                  'up_to_act_ns':b['start_ns']-a['end_ns'], 'act_to_down_ns':c['start_ns']-b['end_ns']}
            pipeline.append(row)
            if layer['ep']==3 and layer['layer_scope_id']=='2128-42':
                selected_f10=[t for t in tasks if t['stream']=='35' and t['type'] in ['MatMulV2','MatMulV3','Cast','Mul']
                              and any(shape==(1023,32) for shape in dims(t['input_shapes'])+dims(t['output_shapes']))]
    assert rank_count==16 and scopes_total==len(pipeline)==1280
    f16=[p for p in pipeline if 'f16_pair' in p]
    f22=[p for p in pipeline if 'f22_chain' in p]
    ep3=next(p['f16_pair'] for p in pipeline if p['ep']==3 and p['scope_id']=='2128-42')
    assert ep3['first']['task_id']=='41440' and ep3['second']['task_id']=='41441'
    provenance=read('agents/fusion/f16-ep3-projection-provenance.json')
    assert len(provenance['rows'])==2
    for label, row in zip(['first','second'],provenance['rows']):
        assert str(row['task']['taskId'])==ep3[label]['task_id']
        assert row['task']['startNs']==ep3[label]['start_ns'] and row['task']['endNs']==ep3[label]['end_ns']
    qkv=read('agents/fusion/f32-ep0-7-projection-rows.json')
    assert len(qkv['rows'])==8 and {r['ep'] for r in qkv['rows']}==set(range(8))
    qrows=[r['tasks']['q_shape']['input_shape'][0] for r in qkv['rows']]
    kvrows=[r['tasks']['kv_shape']['input_shape'][0] for r in qkv['rows']]
    group=read('agents/fusion/f32-ep3-tensor-group.json')
    comm=group['communication_fields']
    same_matrix_input_domain=all(a==b for a,b in zip(qrows,kvrows))
    # Count fields are interpreted conditionally; logical elements are not link/HBM traffic.
    conditional_per_rank_rows=comm['count']//5120 if comm['count']%5120==0 else None
    f16_summary={'paired_scopes':len(f16),'unpaired_scopes':scopes_total-len(f16),
                 'same_stream_pairs':sum(p['f16_pair']['same_stream'] for p in f16),
                 'row_count_matched_pairs':sum(p['f16_pair']['output_input_row_match'] for p in f16),
                 'gap_ns':describe([p['f16_pair']['gap_ns'] for p in f16]),
                 'ep3_exact_host_flow_pair_gap_ns':ep3['gap_ns'],
                 'by_row_count':{str(m):describe([p['f16_pair']['gap_ns'] for p in f16 if dims(p['f16_pair']['second']['input_shapes'])[0][0]==m])
                                  for m in sorted({dims(p['f16_pair']['second']['input_shapes'])[0][0] for p in f16})}}
    f22_summary={'three_task_scopes':len(f22),'other_scopes':scopes_total-len(f22),
                 'same_stream_chains':sum(p['f22_chain']['all_same_stream'] for p in f22),
                 'up_to_act_ns':describe([p['f22_chain']['up_to_act_ns'] for p in f22]),
                 'act_to_down_ns':describe([p['f22_chain']['act_to_down_ns'] for p in f22]),
                 'existing_activation_quantization_composite_observed':True}
    summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'rank_count':rank_count,'scope_count':scopes_total,
             'F06_F32':{'q_rows':qrows,'kv_rows':kvrows,'sum_q_rows':sum(qrows),
                       'same_physical_M_domain':same_matrix_input_domain,
                       'naive_same_input_concatenation_preserving_current_M_domains':'EXCLUDED_FOR_SELECTED_RECORD_SHAPES' if not same_matrix_input_domain else 'NOT_EXCLUDED_BY_SHAPE',
                       'more_general_reorganization':'CONDITIONAL_REQUIRES_ROW_MAPPING_CONSUMERS_NUMERICAL_AND_COMMUNICATION_CONTRACTS',
                       'group_metadata':group['matched_groups'],'group_metadata_limits':group['limits'],
                       'communication_fields':comm,'conditional_rows_per_rank_from_count':conditional_per_rank_rows,
                       'conditional_total_rows_from_equal_size_gather':None if conditional_per_rank_rows is None else conditional_per_rank_rows*comm['rankSize'],
                       'actual_padding_rows_proved':False},
             'F10':{'scope_id':'2128-42','selected_same_shape_task_count':len(selected_f10),'tasks':selected_f10,
                    'meaning':'Projection launch is linked via retained callchains; selected Cast/Mul candidates are shape/stream/time only, not proven tensor edges or scalar values.'},
             'F16':f16_summary,'F22':f22_summary,
             'extraction_audit':audit,'source_sha256':hashlib.sha256(SIDE.read_bytes()).hexdigest(),
             'source_path':str(SIDE),'new_target_source_read':False,'new_device_work':False,
             'whole_modules_closed':0,'overall_complete':False,
             'limits':['Every retained target scope was queried before display. Missing or nonunique candidates remain in detail, not treated as absent execution.',
                       'Scope sidecar is an existing host-flow/CSV extraction with missing exports and selection limits, not the entire raw trace.',
                       'Physical row dimensions are not valid token counts, tensor identity or permission to remove padding.',
                       'Same-stream adjacency establishes exported order, not numerical equivalence or complete consumers.',
                       'Task intervals and gaps under profiler are not launch overhead, critical-path gains or physical utilization.']}
    (OUT/'projection-pipeline-query.json').write_text(json.dumps({'summary':summary,'all_scope_results':pipeline},ensure_ascii=False,indent=2)+'\n')
    (OUT/'projection-pipeline-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    printable={k:summary[k] for k in ['rank_count','scope_count','F06_F32','F16','F22','source_sha256','whole_modules_closed','overall_complete']}
    printable['F10_selected_same_shape_task_count']=len(selected_f10)
    print(json.dumps(printable,ensure_ascii=False))


if __name__=='__main__':
    main()
