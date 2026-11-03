"""Full retained sidecar query for role-specific compressor boundaries."""
from pathlib import Path
from decimal import Decimal
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib,json,statistics

R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
OUT=Path(__file__).parent/'profile-first-20261031'
SIDE=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'


def shapes(text):
    if text in ['N/A','',None]:return []
    return [tuple(int(v) for v in x.split(',')) if x else () for x in text.strip().strip('"').split(';')]


def stats(values):
    return {'n':len(values),'min':min(values),'median':statistics.median(values),'max':max(values)} if values else None


def main():
    cfg=json.loads((R/'evidence/model-config.json').read_text());cfg=cfg.get('text_config',cfg)
    ratios=cfg['compress_ratios'][:cfg['num_hidden_layers']];sources=set(cfg['kv_source_layer_ids'])
    results=[];scope_count=0;all_names=Counter();rank_counts=Counter()
    for line in SIDE.open():
        obj=json.loads(line)
        if obj.get('kind')!='rank':continue
        for scope in obj['data']['layers']:
            scope_count+=1;rank_counts[scope['ep']]+=1
            tasks=[]
            for g in scope['operator_groups']:
                for t in g['tasks']:
                    c=t['csv'];all_names[c['Type']]+=1
                    tasks.append({'type':c['Type'],'name':c['Name'],'task':c['Task ID'],'flow':t['flow_id'],'stream':c['Stream ID'],
                                  'start_ns':int(Decimal(t['device_start_us'])*1000),
                                  'end_ns':int((Decimal(t['device_start_us'])+Decimal(t['trace_duration_us']))*1000),
                                  'input_shapes':c['Input Shapes'],'output_shapes':c['Output Shapes'],
                                  'input_dtypes':c['Input Data Types'],'output_dtypes':c['Output Data Types']})
            pool=[t for t in tasks if 'pool_kernel' in t['name']]
            update=[t for t in tasks if 'cache_update_kernel' in t['name']]
            projections=[t for t in tasks if t['type'] in ['MatMulV2','MatMulV3'] and len(shapes(t['input_shapes']))>=2
                         and len(shapes(t['input_shapes'])[0])==2 and shapes(t['input_shapes'])[0][1]==5120
                         and shapes(t['input_shapes'])[1]==(512,5120)]
            float_proj=[t for t in projections if t['input_dtypes']=='FLOAT;FLOAT']
            if scope['layer'] not in sources and not pool and not update and not float_proj:continue
            norm512=[t for t in tasks if t['type']=='RmsNorm' and shapes(t['input_shapes']) and shapes(t['input_shapes'])[0][-1:]==(512,)]
            keyproj=[t for t in tasks if t['type'] in ['MatMulV2','MatMulV3'] and len(shapes(t['input_shapes']))>=2
                     and len(shapes(t['input_shapes'])[0])==2 and shapes(t['input_shapes'])[0][1]==512
                     and shapes(t['input_shapes'])[1]==(128,512)]
            casts=[t for t in tasks if t['type']=='Cast' and t['input_dtypes']=='DT_BF16' and t['output_dtypes']=='FLOAT'
                   and shapes(t['input_shapes']) and shapes(t['input_shapes'])[0][-1:]==(5120,)]
            cache_writes=[t for t in tasks if t['type']=='ScatterNdUpdateSk' or 'cache' in t['name'].lower()]
            item={'ep':scope['ep'],'layer':scope['layer'],'scope_id':scope['layer_scope_id'],'digest':scope['scheduler_digest'],
                  'configured_kv_source':scope['layer'] in sources,'configured_ratio':ratios[scope['layer']],
                  'projection512_candidates':projections,'float_projection_candidates':float_proj,
                  'pool_candidates':pool,'cache_update_candidates':update,'norm512_candidates':norm512,
                  'key_projection_candidates':keyproj,'BF16_to_FLOAT_hidden_cast_candidates':casts,'cache_write_candidates':cache_writes}
            if len(float_proj)==2:
                a,b=sorted(float_proj,key=lambda x:x['start_ns'])
                item['two_float_projection_relation']={'same_stream':a['stream']==b['stream'],'gap_ns':b['start_ns']-a['end_ns'],
                                                        'same_input_shape':a['input_shapes']==b['input_shapes'],
                                                        'first_duration_ns':a['end_ns']-a['start_ns'],'second_duration_ns':b['end_ns']-b['start_ns']}
            if len(pool)==len(update)==1:
                p,u=pool[0],update[0]
                later=[t for t in norm512 if t['stream']==p['stream'] and t['start_ns']>=p['end_ns']]
                item['pool_update_relation']={'same_stream':p['stream']==u['stream'],'gap_ns':u['start_ns']-p['end_ns'],
                                              'post_pool_same_stream_norm_count':len(later),
                                              'post_pool_same_stream_norms':later,
                                              'valid_compressed_rows_measured':False}
            results.append(item)
    assert scope_count==1280 and len(rank_counts)==16
    c2=[r for r in results if r['configured_kv_source'] and r['configured_ratio']==2]
    c1=[r for r in results if r['configured_kv_source'] and r['configured_ratio']==1]
    def summarize(rows):
        return {'scope_count':len(rows),'layers':sorted({r['layer'] for r in rows}),
                'float_projection_count_hist':dict(Counter(len(r['float_projection_candidates']) for r in rows)),
                'all_projection512_count_hist':dict(Counter(len(r['projection512_candidates']) for r in rows)),
                'pool_count_hist':dict(Counter(len(r['pool_candidates']) for r in rows)),
                'cache_update_count_hist':dict(Counter(len(r['cache_update_candidates']) for r in rows)),
                'key_projection_count_hist':dict(Counter(len(r['key_projection_candidates']) for r in rows)),
                'two_float_projection_gap_ns':stats([r['two_float_projection_relation']['gap_ns'] for r in rows if 'two_float_projection_relation' in r]),
                'pool_update_gap_ns':stats([r['pool_update_relation']['gap_ns'] for r in rows if 'pool_update_relation' in r]),
                'post_pool_norm_count_hist':dict(Counter(r['pool_update_relation']['post_pool_same_stream_norm_count'] for r in rows if 'pool_update_relation' in r))}
    prior=json.loads((R/'agents/fusion/f17-layer2-run2-evidence.json').read_text())
    prior_events=prior['events'];prior_pool=[x for x in prior_events if x['type']=='_pool_kernel'];prior_update=[x for x in prior_events if x['type']=='_cache_update_kernel']
    summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'all_retained_target_scopes_queried':scope_count,'rank_counts':dict(rank_counts),
             'C2_sources':summarize(c2),'C1_sources':summarize(c1),
             'other_role_with_matching_float_or_pool':sum(not r['configured_kv_source'] for r in results),
             'old_decode_sample':{'scope':prior['scope'],'pool_count':len(prior_pool),'cache_update_count':len(prior_update),
                                  'pool_update_gap_ns':prior_update[0]['start_rel_ns']-prior_pool[0]['end_rel_ns'] if len(prior_pool)==len(prior_update)==1 else None,
                                  'scope_not_joined_to_heavy_capture':True},
             'source_path':str(SIDE),'source_sha256':hashlib.sha256(SIDE.read_bytes()).hexdigest(),
             'new_target_source_read':False,'new_device_work':False,'whole_modules_closed':0,'overall_complete':False,
             'limits':['Full retained sidecar scope only; missing/unmatched exports do not imply absent operations.',
                       'Configured role and matching signature are not historical object/tensor/callchain identity.',
                       'Pool/kernel physical row dimensions do not reveal actual complete compressed rows or valid cache slots.',
                       'Same-stream ordering does not prove a universal barrier or forbid tile overlap with disjoint state.',
                       'Old decode and later heavy-capture timings and values remain separate.']}
    (OUT/'compressor-profile-query.json').write_text(json.dumps({'summary':summary,'scope_results':results},ensure_ascii=False,indent=2)+'\n')
    (OUT/'compressor-profile-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
