#!/usr/bin/env python3
"""Bind Attention output -> FFN norm from frozen tasks, then conditional budgets."""
import argparse
import collections
import hashlib
import json
import zlib
from pathlib import Path

from analyze_c2_producer_windows import price_exposure, shape, stats


def extract(archive):
    paths=[archive/'.cache'/n for n in ('first-token-v168-protocol-bodies.json.zlib',
                                      'first-token-v168-collectives.json.zlib')]
    compute,comm=[json.loads(zlib.decompress(p.read_bytes())) for p in paths]
    windows=[]
    for p,c in zip(compute['profiles'],comm['profiles']):
        assert (p['database'],p['sha256'],p['DP'],p['TP']) == (c['database'],c['sha256'],c['DP'],c['TP'])
        for body,cw in zip(p['bodies'],c['windows']):
            assert body['F21_carrier_rows']==cw['carrier_rows']
            by={r[0]:r for r in body['rows']}
            pre=[by[i] for i in body['HcPre_TASK_ROWS']]
            stream,={r[3] for r in pre}
            posts=sorted((r for r in body['rows'] if r[3]==stream
                          and p['regimes'][r[-1]][1]=='HcPost'),key=lambda r:r[1])
            assert len(pre)==len(posts)==80
            for layer in range(40):
                a,b,fp,ffpost=pre[2*layer],posts[2*layer],pre[2*layer+1],posts[2*layer+1]
                rows=[r for r in body['rows'] if a[1]<=r[1] and r[2]<=b[1]]
                def reg(r):return p['regimes'][r[-1]]
                def pick(op):return [r for r in rows if reg(r)[1]==op]
                smla=pick('SparseFlashMla')
                oa=pick('TransposeBatchMatMul')
                ob=[r for r in rows if reg(r)[1].startswith('MatMul')
                    and reg(r)[3]=='DT_BF16;DT_BF16'
                    and shape(reg(r)[2])[1:]==[(5120,1024)]]
                assert len(smla)<=1 and len(oa)==len(ob)<=1
                assert all(reg(r)[3]=='DT_BF16;DT_BF16' for r in oa+ob)
                inverse=[r for r in pick('InplacePartialRotaryMul') if smla and r[1]>=smla[0][2]]
                assert len(inverse)==len(smla)
                pack=[r for r in pick('Transpose') if len(shape(reg(r)[2])[0])==4
                      and shape(reg(r)[2])[0][1:]==(8,8,512)]
                assert len(pack)==len(oa)
                cc=[r for r in cw['communications'] if a[2]<=r[0] and r[1]<=b[1]]
                exchange=[r for r in cc if r[4]=='hcom_alltoall_']
                rs=[r for r in cc if r[4]=='hcom_reduceScatter_']
                assert len(exchange)==len(oa) and len(rs)==1
                norms=[r for r in body['rows'] if fp[2]<=r[1] and r[2]<=ffpost[1]
                       and reg(r)[1]=='RmsNormCast']
                assert len(norms)==1
                def native(r):return dict(task=r[0],us=(r[2]-r[1])/1000,regime=reg(r))
                def dur(v):return sum(r[2]-r[1] for r in v)/1000
                windows.append(dict(dp=p['DP'],tp=p['TP'],carrier=body['F21_carrier_rows'],
                    layer=layer,hc_task=a[0],hc_local_rows=shape(reg(fp)[2])[0][0],
                    branch='query' if smla else 'no_local_query_with_projection' if oa else 'no_projection',
                    smla=native(smla[0]) if smla else None,
                    inverse=[native(r) for r in inverse],pack=[native(r) for r in pack],
                    oa=native(oa[0]) if oa else None,ob=native(ob[0]) if ob else None,
                    exchange=dict(name=exchange[0][3],count_per_peer=exchange[0][7],dtype=exchange[0][8],
                                  us=(exchange[0][1]-exchange[0][0])/1000) if exchange else None,
                    reduce_scatter=dict(name=rs[0][3],count_per_rank=rs[0][7],dtype=rs[0][8],
                                        us=(rs[0][1]-rs[0][0])/1000),
                    hc_post=native(b),ffn_hc_pre=native(fp),ffn_norm=native(norms[0]),
                    inverse_pack_us=dur(inverse+pack),
                    hc_boundary_us=(b[2]-b[1]+fp[2]-fp[1]+norms[0][2]-norms[0][1])/1000))
    arrival_path=archive/'full-review/PREFILL-OA-JOINT-ARRIVAL.json'
    arrival=json.loads(arrival_path.read_text())
    data=dict(windows=windows,carrier_order=[s['M'] for s in arrival['sections']],
              requests=[{k:r[k] for k in ('slot','DP','first_eager_M','TTFT_ms')} for r in arrival['requests']],
              arrival_sections=arrival['sections'],
              source_pins=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths+[arrival_path]])
    return data


def price(data):
    fastest={}
    for w in data['windows']:
        if w['exchange']:
            key=(w['dp'],w['carrier'])
            fastest[key]=min(fastest.get(key,float('inf')),w['exchange']['us'])
    def budget(w,route):
        if route=='OA_before_exchange_ideal_4x':return .75*(w['exchange']['us'] if w['exchange'] else 0)
        if route.startswith('OA_4x_minDuration_proxy_h'):
            if not w['exchange']:return 0
            h=float(route.removeprefix('OA_4x_minDuration_proxy_h'))
            # Same-cohort same-payload native minimum as a scenario anchor,
            # not proof of the byte-sensitive term. Tiny calibration is only
            # a startup proxy, not the new transport's measured latency.
            alpha=(28.6625,28.72)[w['dp']]
            return max(0,.75*max(0,fastest[w['dp'],w['carrier']]-alpha)-h)
        if route=='inverse_pack_entirely_free':return w['inverse_pack_us']
        if route=='Post_Pre_norm_entirely_free':return w['hc_boundary_us']
        raise AssertionError(route)
    scenarios={}
    for route in ('OA_before_exchange_ideal_4x','OA_4x_minDuration_proxy_h0',
                  'OA_4x_minDuration_proxy_h25','OA_4x_minDuration_proxy_h50',
                  'OA_4x_minDuration_proxy_h100','OA_4x_minDuration_proxy_h150',
                  'inverse_pack_entirely_free','Post_Pre_norm_entirely_free'):
        stages=[]
        for section in data['arrival_sections']:
            for layer in range(40):
                groups=[]
                for dp in (0,1):
                    group=[w for w in data['windows'] if (w['dp'],w['carrier'],w['layer'])==(dp,section['M'],layer)]
                    assert sorted(w['tp'] for w in group)==list(range(8))
                    groups.append([dict(tp=w['tp'],branch=route,native_pair_us=budget(w,route)) for w in group])
                stages.append(dict(carrier=section['M'],layer=layer,
                                   DP0_minus_DP1_us=section['layers'][layer]['DP0_minus_DP1_us'],groups=groups))
        params=dict(stage_rows=stages,carrier_order=data['carrier_order'],requests=data['requests'])
        scenarios[route]={k.replace('all_FP32_pairs/remaining','unrealized_'):v
                          for k,v in price_exposure(params).items() if k.startswith('all_FP32_pairs/')}
    groups={}
    for dp,carrier in sorted({(w['dp'],w['carrier']) for w in data['windows']}):
        group=[w for w in data['windows'] if (w['dp'],w['carrier'])==(dp,carrier)]
        groups[f'DP{dp}/carrier{carrier}']=dict(branches=dict(collections.Counter(w['branch'] for w in group)),
            exchange_us=stats([w['exchange']['us'] for w in group if w['exchange']]),
            OA_us=stats([w['oa']['us'] for w in group if w['oa']]),
            OB_us=stats([w['ob']['us'] for w in group if w['ob']]),
            inverse_pack_us=stats([w['inverse_pack_us'] for w in group]),
            hc_boundary_us=stats([w['hc_boundary_us'] for w in group]))
    return dict(status='ARCHIVED_CONDITIONAL_DELIVERY_BUDGET_NOT_EXPECTED_GAIN',
        groups=groups,scenarios=scenarios,
        all_observed_OA_OB_are_BF16=True,windows=len(data['windows']),
        original_bytes_per_query=64*512*2,OA_output_bytes_per_query=8*1024*2,
        persistent_extra_OA_weight_bytes_per_rank=7*4096*1024*2*40,
        serial_on_demand_weight_AG_receive_bytes_per_layer=7*4096*1024*2,
        assumptions=['Retain original EP arrival intervals and external schedule; grant each TP group the largest removable rank budget.',
                     'OA-before-exchange scenario scales entire original AlltoAll duration linearly by payload (1/4), with no startup floor or added OA/weight/copy/contending cost. This is optimistic, not a measured or hard physical bound.',
                     'minDuration proxy uses the same-cohort group/payload minimum, tiny-collective startup proxy DP0 28.6625us/DP1 28.72us, and extra cost h per layer. Neither byte sensitivity nor new OA timing is measured; no on-demand weight transfer included.',
                     'Free epilogue and HC scopes zero all their native work, including necessary MACs/reductions. They are search budgets, not realizable algorithms.',
                     'Routes overlap and are not additive; no service TTFT prediction or broad workload inference.'])


def compact(data):
    names=['query','no_local_query_with_projection','no_projection']
    return dict(schema='attention-output-portable-v1',
        row_fields=['dp','tp','carrier','layer','branch_index','exchange_us','OA_us','OB_us',
                    'inverse_pack_us','hc_boundary_us'],branch_names=names,
        rows=[[w['dp'],w['tp'],w['carrier'],w['layer'],names.index(w['branch']),
               w['exchange']['us'] if w['exchange'] else None,
               w['oa']['us'] if w['oa'] else None,w['ob']['us'] if w['ob'] else None,
               w['inverse_pack_us'],w['hc_boundary_us']] for w in data['windows']],
        requests=data['requests'],carrier_order=data['carrier_order'],source_pins=data['source_pins'],
        arrival_sections=[dict(M=s['M'],layers=[dict(DP0_minus_DP1_us=l['DP0_minus_DP1_us'])
                                               for l in s['layers']]) for s in data['arrival_sections']])


def expand(data):
    if data.get('schema')!='attention-output-portable-v1':return data
    def optional(v):return dict(us=v) if v is not None else None
    return dict(requests=data['requests'],carrier_order=data['carrier_order'],
        source_pins=data['source_pins'],arrival_sections=data['arrival_sections'],
        windows=[dict(dp=r[0],tp=r[1],carrier=r[2],layer=r[3],branch=data['branch_names'][r[4]],
                      exchange=optional(r[5]),oa=optional(r[6]),ob=optional(r[7]),
                      inverse_pack_us=r[8],hc_boundary_us=r[9]) for r in data['rows']])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path)
    ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--compact-output',type=Path)
    ap.add_argument('--recompute',action='store_true');a=ap.parse_args()
    data=expand(json.loads(a.inputs.read_text())) if a.recompute else extract(a.archive)
    if not a.recompute:a.inputs.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    if a.compact_output:a.compact_output.write_text(json.dumps(compact(data),ensure_ascii=False)+'\n')
    result=price(data);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(groups=result['groups'],max_conditional_TTFT_percent={k:v['unrealized_0']['maximum_request_percent'] for k,v in result['scenarios'].items()})))
