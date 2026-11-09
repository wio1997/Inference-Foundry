#!/usr/bin/env python3
"""Bind MoE input/readers and post-DFC shared work. No stage-idle inference."""
import argparse
import collections
import hashlib
import json
import statistics
import zlib
from pathlib import Path
from analyze_c2_producer_windows import shape,stats


def analyze(root):
    p=root/'.cache/first-token-v168-protocol-bodies.json.zlib'
    data=json.loads(zlib.decompress(p.read_bytes()))
    windows=[]
    for prof in data['profiles']:
        for body in prof['bodies']:
            by={r[0]:r for r in body['rows']}
            pres=[by[i] for i in body['HcPre_TASK_ROWS']]
            stream,={r[3] for r in pres}
            posts=sorted((r for r in body['rows'] if r[3]==stream
                          and prof['regimes'][r[-1]][1]=='HcPost'),key=lambda r:r[1])
            assert len(pres)==len(posts)==80
            for layer in range(40):
                pre,post=pres[2*layer+1],posts[2*layer+1]
                rows=[r for r in body['rows'] if pre[2]<=r[1] and r[2]<=post[1]]
                def reg(r):return prof['regimes'][r[-1]]
                def pick(op):return [r for r in rows if reg(r)[1]==op]
                norm,=pick('RmsNormCast');dfc,=pick('DispatchFFNCombine')
                topk,=pick('MoeGatingTopKHash')
                dq,=pick('DynamicQuant');act,=pick('DequantSwigluQuant');add,=pick('ForeachAddList')
                up,=[r for r in rows if reg(r)[1].startswith('QuantBatchMatmul') and reg(r)[5]=='INT32']
                down,=[r for r in rows if reg(r)[1].startswith('QuantBatchMatmul') and reg(r)[5]=='DT_BF16']
                gate,=[r for r in rows if reg(r)[1].startswith('MatMul') and reg(r)[3]=='FLOAT;FLOAT'
                       and shape(reg(r)[2])[1:]==[(384,5120)]]
                local=shape(reg(norm)[2])[0][0]
                assert shape(reg(gate)[2])[0]==(local,5120)
                assert shape(reg(dq)[2])[0]==(local,5120)
                assert shape(reg(up)[4])==[(local,4608)]
                assert shape(reg(down)[4])==[(local,5120)]
                assert norm[2]<=gate[1]<=gate[2]<=topk[1]<=topk[2]<=dfc[1]
                assert dfc[2]<=dq[1]<=dq[2]<=up[1]<=up[2]<=act[1]<=act[2]<=down[1]<=down[2]<=add[1]
                helpers=[r for r in rows if dfc[2]<=r[1] and r[2]<=dq[1]]
                def record(r):return dict(task=r[0],us=(r[2]-r[1])/1000,regime=reg(r))
                windows.append(dict(dp=prof['DP'],tp=prof['TP'],carrier=body['F21_carrier_rows'],layer=layer,
                    local_norm_shared_rows=local,DFC_rows=shape(reg(dfc)[2])[0][0],
                    norm=record(norm),gate=record(gate),topk=record(topk),dfc=record(dfc),
                    shared_DQ=record(dq),shared_Up=record(up),shared_activation=record(act),
                    shared_Down=record(down),combine_add=record(add),
                    post_DFC_diagnostic_observer_tasks=[record(r) for r in helpers],
                    helper_native_us=sum(r[2]-r[1] for r in helpers)/1000,
                    shared_pipeline_native_us=sum(r[2]-r[1] for r in (dq,up,act,down))/1000))
    arrival=root/'full-review/PREFILL-OA-JOINT-ARRIVAL.json'
    reqs=json.loads(arrival.read_text())
    order=[s['M'] for s in reqs['sections']]
    budgets={}
    for field in ('shared_Up','shared_pipeline_native_us'):
        carriers={m:0. for m in order}
        for m in order:
            for layer in range(40):
                selected=[w for w in windows if (w['carrier'],w['layer'])==(m,layer)]
                assert len(selected)==16
                vals=[w[field]['us'] if isinstance(w[field],dict) else w[field] for w in selected]
                carriers[m]+=max(vals)/1000
        requests=[]
        for r in reqs['requests']:
            last=order.index(r['first_eager_M'])+1
            credit=sum(carriers[m] for m in order[:last])
            requests.append(dict(slot=r['slot'],TTFT_ms=r['TTFT_ms'],first_eager_M=r['first_eager_M'],
                                 optimistic_serial_credit_ms=credit,percent=100*credit/r['TTFT_ms']))
        budgets[field]=dict(carriers_ms=carriers,requests=requests,max_percent=max(r['percent'] for r in requests))
    groups={}
    for dp,m in sorted({(w['dp'],w['carrier']) for w in windows}):
        ws=[w for w in windows if (w['dp'],w['carrier'])==(dp,m)]
        groups[f'DP{dp}/carrier{m}']=dict(local_rows=sorted({w['local_norm_shared_rows'] for w in ws}),
            DFC_rows=sorted({w['DFC_rows'] for w in ws}),
            metrics={k:stats([w[k]['us'] if isinstance(w[k],dict) else w[k] for w in ws])
                     for k in ('gate','topk','dfc','shared_DQ','shared_Up','shared_activation','shared_Down',
                               'combine_add','helper_native_us','shared_pipeline_native_us')})
    return dict(status='SOURCE_SELECTED_MOE_READERS_AND_OPTIMISTIC_SERIAL_BUDGET',windows=windows,groups=groups,
        conditional_budgets=budgets,source_pins=[dict(path=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for q in (p,arrival)],
        limits=['At each source-layer/carry cohort grant all ranks the largest original selected work duration, ignoring every EP wait, new overhead and resource conflict; external request/carrier order fixed.',
                'Post-DFC work belongs after the current layer EP arrival. Do not reuse the pre-EP current-layer arrival-credit model for it.',
                'Free-work/fully-hidden serial credit is not expected TTFT or a universal ceiling under arbitrary changed schedules.',
                'The whole DFC native duration reveals neither initial Cube wait nor tail slack. Prefix/tail Up placement are alternatives, never additive.',
                'Post-DFC mask-sum/cat/history-scatter matches the pinned v168 diagnostic observer, not production EPLB. It receives zero model speedup credit.',
                'This is a diagnostic cohort. Observer allocation, reads and scheduling may perturb other native work; subtracting its durations does not reconstruct measured production timing.'],
        new_NPU_measurements=0,whole_chain_complete=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=analyze(a.archive)
    raw=(json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode()
    a.output.write_bytes(zlib.compress(raw,9) if a.output.suffix=='.zlib' else raw)
    print(json.dumps(dict(windows=len(d['windows']),groups=d['groups'],budgets={k:v['max_percent'] for k,v in d['conditional_budgets'].items()})))
