#!/usr/bin/env python3
"""Consumer-derived Engram boundaries, typed archived costs and scoped budgets."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import zlib
from analyze_c2_producer_windows import price_exposure, shape, stats

BASE = Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT = Path('audit/execution-search')
PINS = {}


def read(name):
    p = BASE/name
    raw = p.read_bytes()
    PINS[str(p)] = hashlib.sha256(raw).hexdigest()
    return json.loads(zlib.decompress(raw) if p.suffix == '.zlib' else raw)


def main():
    compute = read('.cache/first-token-v168-protocol-bodies.json.zlib')
    units = read('full-review/PROFILE-UNIT-COSTS.json')
    arrival = read('full-review/PREFILL-OA-JOINT-ARRIVAL.json')
    decode = read('full-review/CURRENT-MECHANISM-VALUE.json')
    export = read('full-review/runs/publication-residual-profile-cpu-plan-2979e908408a-export-result.json')
    ws = []
    wkv_groups = defaultdict(list)
    signatures = []
    expected_gate_ops = ['Cast','Cast','Mul','Cast','Cast','MatMulV2','Slice','Cast',
        'Pows','ReduceMean','Add','Rsqrt','Pows','ReduceMean','Add','Rsqrt','Mul','Mul',
        'Mul','ReduceSum','Mul','Mul','Abs','ClipByValueV2','Sqrt','Less','Neg','SelectV2',
        'Sigmoid','LogicalNot','MaskedFill','Cast','Slice','Cast','Mul','Add','Cast']

    def sig(g):
        if g not in signatures:
            signatures.append(g)
        return signatures.index(g)

    def us(rows):
        intervals = sorted((r[1], r[2]) for r in rows)
        end = -1
        total = 0
        for a, b in intervals:
            total += max(0, b-max(a, end))
            end = max(end, b)
        return total/1000

    for p in compute['profiles']:
        for body in p['bodies']:
            by = {r[0]:r for r in body['rows']}
            pre = [by[i] for i in body['HcPre_TASK_ROWS']]
            stream, = {r[3] for r in pre}
            posts = sorted((r for r in body['rows'] if r[3] == stream
                and p['regimes'][r[-1]][1] == 'HcPost'), key=lambda r:r[1])
            assert len(pre) == len(posts) == 80
            for layer in [1, 14]:
                left, right = posts[2*layer-1], pre[2*layer]
                rows = sorted((r for r in body['rows'] if r[3] == stream
                    and left[2] <= r[1] and r[2] <= right[1]), key=lambda r:r[1])
                kv, = [r for r in rows if p['regimes'][r[-1]][1] == 'MatMulV2'
                    and '25600,6144' in p['regimes'][r[-1]][2]]
                gate = rows[rows.index(kv)+1:rows.index(kv)+38]
                assert [p['regimes'][r[-1]][1] for r in gate] == expected_gate_ops
                reg = p['regimes'][kv[-1]]
                assert reg[3] == 'DT_BF16;DT_BF16' and reg[5] == 'DT_BF16'
                m = shape(reg[2])[0][0]
                assert shape(reg[2])[1] == (25600,6144)
                assert shape(p['regimes'][right[-1]][2])[0] == (m,4,5120)
                tuple_rows = [kv]+[gate[i] for i in [6,7,12,13,14,15,32,33]]
                immutable = [gate[i] for i in [0,1,2,4,31]]
                ws.append(dict(dp=p['DP'],tp=p['TP'],carrier=body['F21_carrier_rows'],layer=layer,
                    local_row_capacity=m, previous_post_task=left[0], next_pre_task=right[0],
                    wkv_us=us([kv]),gate_us=us(gate),body_us=us(rows),
                    tuple_us=us(tuple_rows),immutable_and_duplicate_cast_us=us(immutable),
                    body_and_next_pre_us=us(rows+[right]),
                    typed_rows=[[r[0],(r[1]-left[2])/1000,(r[2]-left[2])/1000,
                                 sig(p['regimes'][r[-1]])] for r in rows]))
                wkv_groups['eager_first_token',p['DP'],body['F21_carrier_rows'],tuple(reg)].append(us([kv]))
    assert len(ws) == 128

    decode_rows = []
    for pin in export['local_cache']:
        p = Path(pin['path']);raw=p.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pin['sha256']
        PINS[str(p)] = pin['sha256']
        data = json.loads(zlib.decompress(raw) if p.suffix == '.zlib' else raw)
        by = {r[0]:r for r in data['rows']}
        for w in decode['windows']:
            if w['rank'] != data['rank']:
                continue
            for task in w['groups']['Engram_WKV']['TASK_ROW_ids']:
                r = by[task];g=data['regimes'][r[9]]
                reg=[g[k] for k in ['name','opType','inputShapes','inputDataTypes','outputShapes','outputDataTypes']]
                assert shape(reg[2]) == [(3,6144),(25600,6144)]
                wkv_groups['target_graph',data['DP'],None,tuple(reg)].append(us([r]))
            decode_rows.append(dict(rank=w['rank'],ordinal=w['ordinal'],period_ms=w['period_ms'],
                groups={k:w['groups'][k] for k in ['Engram_WKV','Engram_rotated_gate_all',
                    'Engram_immutable_constants','Engram_lookup_zero_transpose',
                    'Engram_gate_and_direct_lookup','Engram_feature_collectives_including_wait']}))
    assert len(decode_rows) == 112
    matches = []
    for (phase,dp,carrier,reg),values in wkv_groups.items():
        s, = [s for s in units['signatures'] if s['phase']==phase and
            [s[k] for k in ['name','opType','inputShapes','inputDataTypes','outputShapes','outputDataTypes']]==list(reg)]
        c, = [c for c in s['conditions'] if (c['dp'],c['carrier'])==(dp,carrier)]
        assert c['calls'] == len(values)
        for k,v in [('min',min(values)),('median',statistics.median(values)),('max',max(values))]:
            assert abs(c['observed_us'][k]-v)<1e-8
        matches.append(dict(phase=phase,dp=dp,carrier=carrier,signature=list(reg),calls=len(values),us=stats(values)))
    assert len(matches) == 10

    prices = {}
    for field in ['wkv_us','gate_us','body_us','tuple_us','immutable_and_duplicate_cast_us','body_and_next_pre_us']:
        stages = []
        for s in arrival['sections']:
            for layer in [1,14]:
                gs = [[dict(tp=w['tp'],native_pair_us=w[field],branch='engram') for w in ws
                    if (w['dp'],w['carrier'],w['layer'])==(dp,s['M'],layer)] for dp in [0,1]]
                assert all(sorted(w['tp'] for w in g)==list(range(8)) for g in gs)
                stages.append(dict(carrier=s['M'],layer=layer,
                    DP0_minus_DP1_us=s['layers'][layer]['DP0_minus_DP1_us'],groups=gs))
        price = price_exposure(dict(stage_rows=stages,carrier_order=[s['M'] for s in arrival['sections']],
            requests=[{k:r[k] for k in ['slot','DP','first_eager_M','TTFT_ms']} for r in arrival['requests']]))
        prices[field] = {k.replace('all_FP32_pairs/remaining','remaining_'):v for k,v in price.items() if k.startswith('all_FP32_pairs/')}

    group_summary = []
    for dp,carrier in sorted({(w['dp'],w['carrier']) for w in ws}):
        sel = [w for w in ws if (w['dp'],w['carrier'])==(dp,carrier)]
        group_summary.append(dict(dp=dp,carrier=carrier,local_row_capacity=sorted({w['local_row_capacity'] for w in sel}),
            **{f:stats([w[f] for w in sel]) for f in ['wkv_us','gate_us','body_us','tuple_us']}))
    sizes = {}
    for m in sorted({w['local_row_capacity'] for w in ws}|{3}):
        sizes[m]=dict(local_rows=m,full_TP_padded_row_capacity=8*m,
            original_feature_bank_B=8*m*6144*2,SP_owner_bank_B=m*6144*2,
            old_rank_receive_payload_B=7*m*6144*2,new_owner_rank_receive_payload_B=7*m*3*256*2,
            projected_BF16_KV_B=m*25600*2,projected_KV_and_key_rstd_B=m*(25600*2+4*4),
            FP32_KV_and_key_rstd_B=m*(25600*4+4*4),mandatory_post_gate_BF16_residual_B=m*4*5120*2)
    result=dict(status='ENGRAM_CONSUMER_PRODUCER_REVIEW_PARTIAL_NO_NATIVE_CANDIDATE',
        counts=dict(first_token_body_intervals=64,Engram_boundaries=len(ws),gate_tasks=37*len(ws),
                    WKV_cost_groups_exactly_matched=len(matches),decode_periods=len(decode_rows)),
        WKV_cost_groups=matches,first_token_group_summary=group_summary,
        first_token_conditional_free_work=prices,logical_delivery_and_bank_sizes=sizes,
        source_profile_pins=PINS,gate_task_ops=expected_gate_ops,
        decode_scope='Separate current captured M3 cohort; do not add its periods to C8 first-token budgets.',
        limits=['Budgets preserve original external order/EP arrival intervals and optimistically grant maximum rank credit to all TP peers; they are scoped conditional search budgets, not expected request gains or a bound on changed scheduling.',
            'Free body includes required BF16 WKV/rotation/norm/gate/residual computation and boundary copies. Lookup, TP gather and preparation host work are outside this body. Body+nextPre adds the immediate HcPre only; residual still has a later HcPost reader.',
            'A local row capacity or padded TP bank size does not establish actual valid/hash-hit/mask population.',
            'No times across unrelated cohorts are added. Collective wait duration is not intrinsic transit; byte savings need a different transport cost.',
            'Content misses compacted into a different M/N projection have no bitwise equivalence proof. Cache keys include arithmetic geometry, weights, lookup codes/scales, table/head/owner and state generations.',
            'All alternatives overlap; cost groups prove archived typed call binding, not runtime physical formats or new candidate value.'],
        new_NPU_measurements=0,service_requests=0,whole_chain_complete=False)
    (OUT/'engram-execution-frontier.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    portable=dict(schema='engram-consumer-windows-v1',signature_fields=['name','opType','inputShapes','inputDataTypes','outputShapes','outputDataTypes'],
        task_fields=['TASK_ROW','start_from_previous_post_end_us','end_from_previous_post_end_us','signature_index'],
        signatures=signatures,windows=ws,decode_windows=decode_rows)
    (OUT/'engram-execution-windows.json').write_text(json.dumps(portable,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(dict(counts=result['counts'],max_conditional_request_percent={k:v['remaining_0']['maximum_request_percent'] for k,v in prices.items()},bank_sizes=sizes),ensure_ascii=False))


if __name__ == '__main__':
    main()
