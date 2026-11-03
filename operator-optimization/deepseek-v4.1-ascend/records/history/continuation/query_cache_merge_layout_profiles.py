from pathlib import Path
from decimal import Decimal
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,statistics

R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
OUT=Path(__file__).parent/'profile-first-20261031'
SIDE=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'


def shp(s):
    if not s or s=='N/A':return []
    return [tuple(int(v.strip()) for v in a.split(',')) if a.strip() else () for a in s.strip().strip('"').split(';')]


def stat(v):
    return {'n':len(v),'min':min(v),'median':statistics.median(v),'max':max(v)} if v else None


def main():
    cfg=json.loads((R/'evidence/model-config.json').read_text());cfg=cfg.get('text_config',cfg)
    sources=set(cfg['kv_source_layer_ids']);rows=[];targets=[]
    for line in SIDE.open():
        data=json.loads(line)
        if data.get('kind')!='rank':continue
        for s in data['data']['layers']:
            tasks=[]
            for group in s['operator_groups']:
                for a in group['tasks']:
                    c=a['csv'];start=int(Decimal(a['device_start_us'])*1000);duration=int(Decimal(a['trace_duration_us'])*1000)
                    tasks.append({'type':c['Type'],'name':c['Name'],'task_id':c['Task ID'],'flow':a['flow_id'],'stream':c['Stream ID'],
                                  'start_ns':start,'end_ns':start+duration,'duration_ns':duration,
                                  'host_start_us':a['host_start_us'],'host_pid':a['host_pid'],'host_tid':a['host_tid'],
                                  'input_shapes':c['Input Shapes'],'output_shapes':c['Output Shapes'],
                                  'input_dtypes':c['Input Data Types'],'output_dtypes':c['Output Data Types']})
            tasks.sort(key=lambda x:x['start_ns'])
            base={'ep':s['ep'],'layer':s['layer'],'scope_id':s['layer_scope_id'],'digest':s['scheduler_digest']}
            merge=[t for t in tasks if t['type']=='ForeachAddList']
            post=[t for t in tasks if t['type']=='HcPost']
            routed=[t for t in tasks if t['type']=='DispatchFFNCombine']
            down=[t for t in tasks if t['type']=='QuantBatchMatmulV3' and shp(t['input_shapes']) and shp(t['input_shapes'])[0][-1:]==(2304,) and shp(t['output_shapes'])[0][-1:]==(5120,)]
            row={**base,'merge_count':len(merge),'hcpost_count':len(post),'routed_count':len(routed),'shared_down_count':len(down)}
            if len(merge)==len(routed)==len(down)==1 and len(post)==2:
                m=merge[0];h=max(post,key=lambda t:t['start_ns'])
                row['merge_chain']={'routed':routed[0],'shared_down':down[0],'merge':m,'last_hcpost':h,
                                    'merge_to_hcpost_gap_ns':h['start_ns']-m['end_ns'],
                                    'shared_down_end_to_merge_start_ns':m['start_ns']-down[0]['end_ns'],
                                    'routed_end_to_merge_start_ns':m['start_ns']-routed[0]['end_ns']}
            sm=[t for t in tasks if t['type']=='SparseFlashMla']
            tb=[t for t in tasks if t['type']=='TransposeBatchMatMul']
            row['attention_count']=len(sm);row['output_projection_first_count']=len(tb)
            if len(sm)==len(tb)==1:
                a,b=sm[0],tb[0]
                between=[t for t in tasks if t['start_ns']>=a['end_ns'] and t['end_ns']<=b['start_ns']]
                rope=[t for t in between if t['type']=='InplacePartialRotaryMul' and shp(t['input_shapes']) and shp(t['input_shapes'])[0][-1:]==(512,)]
                trans=[t for t in between if t['type']=='Transpose']
                comm=[t for t in between if t['type'].startswith('hcom_')]
                row['post_attention_layout_candidates']={'attention':a,'output_projection':b,'rope':rope,'transpose':trans,'communication':comm,
                                                          'interval_ns':b['start_ns']-a['end_ns'],'same_tensor_edges_proved':False}
            if s['layer'] in sources:
                key=[t for t in tasks if t['type'] in ['MatMulV2','MatMulV3'] and len(shp(t['input_shapes']))>=2 and shp(t['input_shapes'])[1]==(128,512)]
                row['index_key_projection_count']=len(key)
                if len(key)==1:
                    k=key[0];m=shp(k['input_shapes'])[0][0]
                    candidates=[t for t in tasks if t['start_ns']>=k['end_ns'] and t['stream']==k['stream']]
                    norm=[t for t in candidates if t['type']=='RmsNorm' and shp(t['input_shapes']) and shp(t['input_shapes'])[0]==(m,128)]
                    rope=[t for t in candidates if t['type']=='InplacePartialRotaryMul' and shp(t['input_shapes']) and shp(t['input_shapes'])[0][-1:]==(128,)]
                    quant=[t for t in candidates if t['type']=='DynamicQuant' and shp(t['input_shapes']) and shp(t['input_shapes'])[0]==(m,128)]
                    keywrite=[t for t in candidates if t['type']=='ScatterNdUpdateSk' and 'INT8' in t['input_dtypes'] and shp(t['input_shapes'])[-1:]==[(m,128)]]
                    scalewrite=[t for t in candidates if t['type']=='ScatterNdUpdateSk' and 'FLOAT16' in t['input_dtypes'] and shp(t['input_shapes'])[-1:]==[(m,1)]]
                    row['key_cache_candidates']={'projection':k,'physical_rows':m,'norm':norm,'rope':rope,'quant':quant,'key_write':keywrite,'scale_write':scalewrite,'tensor_slot_value_identity_proved':False}
            rows.append(row)
            if s['ep']==3 and s['layer_scope_id']=='2128-42' and 'merge_chain' in row:
                for label,key in [('F23_MERGE_L36','merge'),('F23_HCPOST_L36','last_hcpost')]:
                    t=row['merge_chain'][key];targets.append({'label':label,'ep':3,'scope_id':s['layer_scope_id'],**t})
            if s['ep']==3 and s['layer']==2 and s['scheduler_digest'].startswith('db852bff') and 'key_cache_candidates' in row:
                c=row['key_cache_candidates']
                for label,vals in [('F11_KEY_PROJECTION_L2',[c['projection']]),('F11_KEY_NORM_L2',c['norm']),('F11_KEY_QUANT_L2',c['quant']),('F11_KEY_WRITE_L2',c['key_write']),('F11_SCALE_WRITE_L2',c['scale_write'])]:
                    if len(vals)==1:targets.append({'label':label,'ep':3,'scope_id':s['layer_scope_id'],**vals[0]})
    assert len(rows)==1280
    merge_rows=[r for r in rows if 'merge_chain' in r];layout=[r['post_attention_layout_candidates'] for r in rows if 'post_attention_layout_candidates' in r]
    cache=[r['key_cache_candidates'] for r in rows if 'key_cache_candidates' in r]
    summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'retained_scopes':len(rows),
             'F23':{'chain_count':len(merge_rows),'merge_count_hist':dict(Counter(r['merge_count'] for r in rows)),
                    'hcpost_count_hist':dict(Counter(r['hcpost_count'] for r in rows)),
                    'merge_to_last_hcpost_gap_ns':stat([r['merge_chain']['merge_to_hcpost_gap_ns'] for r in merge_rows]),
                    'merge_type_seen':'ForeachAddList','merge_semantics_not_inferred_from_name':True},
             'F14_F15':{'bounded_layout_intervals':len(layout),'post_attention_rope_counts':dict(Counter(len(r['rope']) for r in layout)),
                        'transpose_counts':dict(Counter(len(r['transpose']) for r in layout)),
                        'communication_counts':dict(Counter(len(r['communication']) for r in layout)),
                        'communication_type_counts':dict(Counter(t['type'] for r in layout for t in r['communication']))},
             'F11':{'source_scopes_with_key_projection':len(cache),'candidate_count_hists':{k:dict(Counter(len(r[k]) for r in cache)) for k in ['norm','rope','quant','key_write','scale_write']},
                    'physical_key_projection_rows':dict(Counter(r['physical_rows'] for r in cache))},
             'selected_DB_target_labels':[t['label'] for t in targets],
             'source_sha256':hashlib.sha256(SIDE.read_bytes()).hexdigest(),'source_path':str(SIDE),
             'new_target_source_read':False,'new_device_work':False,'whole_modules_closed':0,'overall_complete':False,
             'limits':['All retained scope rows queried; subset extraction misses exports and does not establish absence outside it.',
                       'Same shape/stream/order and configured role do not prove tensor, slot, cache version or collective ownership.',
                       'ForeachAddList label is not a proof of routed/shared scaling, tuple return or outer model branch.',
                       'Output layout and communication candidates are interval selections, not a proven causal chain.',
                       'Task gaps and sums are not physical utilization or removable request time.']}
    (OUT/'cache-merge-layout-query.json').write_text(json.dumps({'summary':summary,'scope_results':rows},ensure_ascii=False,indent=2)+'\n')
    (OUT/'cache-merge-layout-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'cache-merge-db-targets.json').write_text(json.dumps({'targets':targets,'scope':'exact selected EP3 sidecar anchors for bounded archived DB queries','overall_complete':False},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
