#!/usr/bin/env python3
"""Price a source-selected AG/Wkv pipeline against matching archived EP arrivals.

No kernel timing is predicted. The free-Wkv case is an optimistic, conditional
schedule credit: all other work and original arrival intervals remain fixed.
"""
import argparse
import hashlib
import json
import statistics
import zlib
from pathlib import Path

from analyze_c2_producer_windows import price_exposure, shape, stats


def extract(archive):
    paths = [archive / '.cache' / n for n in (
        'first-token-v168-protocol-bodies.json.zlib',
        'first-token-v168-collectives.json.zlib')]
    compute, comm = [json.loads(zlib.decompress(p.read_bytes())) for p in paths]
    windows = []
    for p, c in zip(compute['profiles'], comm['profiles']):
        assert (p['database'], p['sha256'], p['DP'], p['TP']) == (
            c['database'], c['sha256'], c['DP'], c['TP'])
        for body, cw in zip(p['bodies'], c['windows']):
            assert body['F21_carrier_rows'] == cw['carrier_rows']
            by = {r[0]: r for r in body['rows']}
            pre = [by[i] for i in body['HcPre_TASK_ROWS']]
            stream, = {r[3] for r in pre}
            posts = sorted((r for r in body['rows'] if r[3] == stream
                            and p['regimes'][r[-1]][1] == 'HcPost'), key=lambda r: r[1])
            assert len(pre) == len(posts) == 80
            for layer in range(40):
                a, b = pre[layer * 2], posts[layer * 2]
                rows = [r for r in body['rows'] if a[1] <= r[1] and r[2] <= b[1]]
                local = shape(p['regimes'][a[-1]][2])[0][0]
                gathers = [r for r in cw['communications'] if a[2] <= r[0]
                           and r[1] <= b[1] and r[4] == 'hcom_allGather_']
                assert len(gathers) == 1
                ag, = gathers
                assert ag[7:9] == [local * 5120, 8]
                def pick(n):
                    return [r for r in rows if p['regimes'][r[-1]][1].startswith('MatMul')
                            and p['regimes'][r[-1]][3] == 'DT_BF16;DT_BF16'
                            and shape(p['regimes'][r[-1]][2])[1:] == [(n, 5120)]]
                kv = sorted(pick(512), key=lambda r: r[1])
                qa = pick(1280)
                assert len(qa) <= 1
                # C1 source 20 additionally projects long-KV latent, after SWA.
                assert len(kv) == (2 if layer == 20 else 1) if kv else not qa
                global_m = shape(p['regimes'][kv[0][-1]][2])[0][0] if kv else 0
                q_m = shape(p['regimes'][qa[0][-1]][2])[0][0] if qa else 0
                assert q_m == max(0, min(local, global_m - p['TP'] * local))
                def task(r):
                    return dict(task_row=r[0], start_us=(r[1]-a[1])/1000,
                                end_us=(r[2]-a[1])/1000, stream=r[3],
                                regime=p['regimes'][r[-1]])
                windows.append(dict(dp=p['DP'], tp=p['TP'], carrier=body['F21_carrier_rows'],
                    layer=layer, hc_task=a[0], hc_local_rows=local, cache_rows=global_m,
                    query_rows=q_m, gathered_rows=local*8,
                    original_AG_us=(ag[1]-ag[0])/1000,
                    original_AG_start_us=(ag[0]-a[1])/1000,
                    original_AG_end_us=(ag[1]-a[1])/1000,
                    query_a=task(qa[0]) if qa else None,
                    BF16_projections=[task(r) for r in kv],
                    SWA_native_us=(kv[0][2]-kv[0][1])/1000 if kv else 0,
                    all_BF16_native_us=sum(r[2]-r[1] for r in kv)/1000,
                    padding_changes_MM_M=bool(global_m and local*8 != global_m),
                    role='C2' if layer in (2,8,14) else 'C1' if layer == 20 else 'SWA'))
    arrival_path = archive / 'full-review/PREFILL-OA-JOINT-ARRIVAL.json'
    arrival = json.loads(arrival_path.read_text())
    stages = []
    for section in arrival['sections']:
        for layer in range(40):
            if layer in (2, 8, 14):
                continue
            original = section['layers'][layer]
            assert original['layer'] == layer
            groups = []
            for dp in (0, 1):
                group = [w for w in windows if (w['dp'],w['carrier'],w['layer'])
                         == (dp,section['M'],layer)]
                assert sorted(w['tp'] for w in group) == list(range(8))
                # Upper credit for AG+SWA-MM pipeline: no cost regression, whole
                # original SWA MM free, even when overlap cannot be full.
                groups.append([dict(tp=w['tp'], branch='pipeline',
                                    native_pair_us=min(w['original_AG_us'],w['SWA_native_us']))
                               for w in group])
            stages.append(dict(carrier=section['M'],layer=layer,
                               DP0_minus_DP1_us=original['DP0_minus_DP1_us'],groups=groups))
    portable = dict(stage_rows=stages,carrier_order=[s['M'] for s in arrival['sections']],
        requests=[{k:r[k] for k in ('slot','DP','first_eager_M','TTFT_ms')}
                  for r in arrival['requests']],
        assumptions=[
            '37 non-C2 source layers; retain original C2 route, C1 second projection, Q and all other native work.',
            'Grant each TP group the largest rank min(original AG, SWA-MM) credit, including cache-only peers; no new operator or contention cost.',
            'Keep original EP arrival intervals and external schedule fixed; cumulative credits are conditional exposure proxies, not measured TTFT or a universal upper bound.',
            'No inherited local-M compute deletion or 7/8 projection speedup.'])
    pins = [dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in paths+[arrival_path]]
    portable['source_pins'] = pins
    grouped = {}
    for dp, carrier in sorted({(w['dp'],w['carrier']) for w in windows}):
        group = [w for w in windows if (w['dp'],w['carrier']) == (dp,carrier)]
        grouped[f'DP{dp}/carrier{carrier}'] = dict(
            windows=len(group),cache_rows=sorted({w['cache_rows'] for w in group}),
            hc_local_rows=sorted({w['hc_local_rows'] for w in group}),
            AG_us=stats([w['original_AG_us'] for w in group]),
            SWA_us=stats([w['SWA_native_us'] for w in group if w['SWA_native_us']]),
            padding_M_mismatch=sum(w['padding_changes_MM_M'] for w in group))
    return dict(status='ARCHIVED_SOURCE_SELECTED_TRANSPORT_BUDGET_NOT_BENCHMARK',
                source_pins=pins,windows=windows,groups=grouped), portable


def price(data):
    all_cases = price_exposure(data)
    cases = {k.replace('all_FP32_pairs','AG_SWA_pipeline'):
             v for k,v in all_cases.items() if k.startswith('all_FP32_pairs/')}
    free = cases['AG_SWA_pipeline/remaining0']
    return dict(scenarios=cases, free_overlap_max_request_percent=free['maximum_request_percent'],
                assumptions=data['assumptions'],
                limits=['remaining is unrealized fraction of optimistic overlap credit, not measured operator latency.',
                        'One percent is the existing prioritization reference, not a new user acceptance rule.'])


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path)
    ap.add_argument('--windows-output',type=Path)
    ap.add_argument('--portable-input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--recompute',action='store_true')
    a=ap.parse_args()
    if a.recompute:
        data=json.loads(a.portable_input.read_text())
    else:
        if not a.archive or not a.windows_output:ap.error('archive and windows-output required')
        windows,data=extract(a.archive)
        a.windows_output.write_text(json.dumps(windows,ensure_ascii=False,indent=2)+'\n')
        a.portable_input.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'groups':windows['groups'],'windows':len(windows['windows'])}))
    output=price(data)
    a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output))
