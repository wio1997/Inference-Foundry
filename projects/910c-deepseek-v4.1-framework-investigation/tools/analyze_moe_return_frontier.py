#!/usr/bin/env python3
"""Price source-derived return ranges; no phase fractions inferred from DFC."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import zlib

BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT=Path('audit/execution-search')


def main():
    delivery=OUT/'moe-delivery-windows.json.zlib'
    ws=json.loads(zlib.decompress(delivery.read_bytes()))['windows']
    protocol=BASE/'.cache/first-token-v168-protocol-bodies.json.zlib'
    data=json.loads(zlib.decompress(protocol.read_bytes()))
    lookup={}
    for prof in data['profiles']:
        for body in prof['bodies']:
            by={r[0]:r for r in body['rows']}
            pres=[by[i] for i in body['HcPre_TASK_ROWS']]
            stream,={r[3] for r in pres}
            posts=sorted((r for r in body['rows'] if r[3]==stream and prof['regimes'][r[-1]][1]=='HcPost'),key=lambda r:r[1])
            assert len(posts)==80
            for layer in range(40):
                post=posts[2*layer+1]
                lookup[prof['DP'],prof['TP'],body['F21_carrier_rows'],layer]=dict(task=post[0],us=(post[2]-post[1])/1000,regime=prof['regimes'][post[-1]])
    groups={}
    for dp,m in sorted({(w['dp'],w['carrier']) for w in ws}):
        sel=[w for w in ws if (w['dp'],w['carrier'])==(dp,m)]
        pad,local=sel[0]['DFC_rows'],sel[0]['local_norm_shared_rows']
        assert all((w['DFC_rows'],w['local_norm_shared_rows'])==(pad,local) for w in sel)
        groups[f'DP{dp}/carrier{m}']=dict(padded_rows=pad,original_local_rows=local,discarded_output_rows=pad-local,avoidable_output_store_bytes_per_layer_rank=(pad-local)*5120*2,merge_and_HcPost_free_median_us=statistics.median(w['combine_add']['us']+lookup[w['dp'],w['tp'],m,w['layer']]['us'] for w in sel))
    arrival=BASE/'full-review/PREFILL-OA-JOINT-ARRIVAL.json'
    req=json.loads(arrival.read_text());order=[v['M'] for v in req['sections']]
    budgets={}
    for kind in ['shared_DQ','merge_HcPost']:
        carriers={m:0. for m in order}
        for m in order:
            for layer in range(40):
                sel=[w for w in ws if (w['carrier'],w['layer'])==(m,layer)]
                assert len(sel)==16
                vals=[w['shared_DQ']['us'] if kind=='shared_DQ' else w['combine_add']['us']+lookup[w['dp'],w['tp'],m,layer]['us'] for w in sel]
                carriers[m]+=max(vals)/1000
        rs=[]
        for r in req['requests']:
            credit=sum(carriers[m] for m in order[:order.index(r['first_eager_M'])+1])
            rs.append(dict(slot=r['slot'],TTFT_ms=r['TTFT_ms'],credit_ms=credit,percent=100*credit/r['TTFT_ms']))
        budgets[kind]=dict(carriers_ms=carriers,requests=rs,max_percent=max(r['percent'] for r in rs))
    small=BASE/'.cache/v168-target-moe-cost-rows.json.zlib'
    ranks=json.loads(zlib.decompress(small.read_bytes()));rows=[r for rank in ranks for r in rank['rows']]
    assert len(ranks)==16 and len(rows)==144000 and {r[7] for r in rows}=={3}
    receiver=dict(scope='historical captured M3 typed MC2 joins, not C8 eager counts or current production frequency',rows=len(rows),ranks=len(ranks),max_receiver_rows=max(r[9] for r in rows),capacity_values=sorted({r[12] for r in rows}),zero_sender_positive_receiver=sum(r[8]==0 and r[9]>0 for r in rows),all_empty_receiver=sum(r[9]==0 for r in rows),empty_expert_slots=sum(24-r[10] for r in rows),total_expert_slots=24*len(rows),witness_zero_sender_inbound=next(dict(rank=rank['rank'],typed_row=r) for rank in ranks for r in rank['rows'] if r[8]==0 and r[9]>0))
    pins=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [delivery,protocol,arrival,small]]
    d=dict(status='SOURCE_RETURN_FRONTIER_PARTIALLY_REVIEWED',prefix_groups=groups,conditional_free_work_budgets=budgets,historical_receiver_controls=receiver,pins=pins,precision_contract=['Routed GMM1/GMM2 CType is FP16, not shared INT32; preserve fixpipe FP16 boundary','CombineV1 and V2: FP16 -> FP32 -> pertoken Muls -> BF16 CAST_RINT before remote publication','Unpermute: widen BF16, FP32 Muls then sequential topK Add, final BF16 CAST_RINT','Shared merge adds two BF16 tensors with BF16 result before HcPost; preserve this boundary','Capacity and route-index bound remain padded M*6 when skipping local trimmed output rows','End barrier precedes Unpermute: peer offsetD remains live until final local read; next-generation count readiness cannot be bypassed'],limits=['Free-work budget holds original observer-cohort request and carrier schedule fixed, ignores all EP waits/new cost/contention; not expected production gain','No per-phase DFC timing inferred; prefix-output skipping, active-expert publication and remote partial returns remain separate pending candidates','zero sender does not permit whole-DFC early return; counts, remote service and generation still required'],new_NPU_work=0,whole_chain_complete=False)
    (OUT/'moe-return-frontier-summary.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(groups=groups,budgets={k:v['max_percent'] for k,v in budgets.items()},receiver=receiver),ensure_ascii=False))


if __name__=='__main__':main()
