#!/usr/bin/env python3
"""Offline original-C8 three-body pricing and exact expert-count joins.

No service, Torch, SQL mutation, remote connection or new device profiling.
All timings include the original observer and original scheduling.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
import zlib

sys.dont_write_bytecode = True
BASE = Path('/Users/wio/work/branch-fusion-audit-20261104')
FULL = BASE / 'full-review'
OUT = Path('audit/execution-search')
sys.path.insert(0, str(BASE / 'tools'))
from price_vocab_selection_fusion import records, cached, RECEIPT
from price_attention_send_epilogue import ENDPOINTS
from price_draft_movable_arrivals import latest_change


def read(p):
    return json.loads(Path(p).read_bytes())


def digest(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def stats(v):
    return dict(n=len(v), min=min(v), median=statistics.median(v), max=max(v))


def main():
    identity = read(OUT / 'embedding-context-execution-identity.json')
    for r in identity['sources']:
        assert digest(r['local_path']) == r['runtime_sha256']
    ctx_receipt = read(OUT / 'context-parent-receipt.json')
    assert not ctx_receipt['unqualified']
    assert digest(ctx_receipt['raw_path']) == ctx_receipt['raw_sha256']
    context = json.loads(zlib.decompress(Path(ctx_receipt['raw_path']).read_bytes()))
    waves, hist, schedule, _, _ = records()
    profiles = cached(RECEIPT)
    crossover_path = FULL / 'F21-SAME-COHORT-CROSSOVER.json'
    crossover = read(crossover_path)
    # Counts are qualified independently of the old failed empty-credit recipe.
    counts_path = BASE / '.cache/saved-v168-f21-expert-counts.json.zlib'
    publication = read(FULL / 'F21-EMPTY-EXPERT-PUBLICATION.json')
    assert digest(counts_path) == publication['inputs'][str(counts_path)]
    counts = json.loads(zlib.decompress(counts_path.read_bytes()))
    crossover_pins = []
    for p, expected in crossover['reused_input_sha256'].items():
        actual = digest(p)
        assert actual == expected, p
        crossover_pins.append(dict(path=p, sha256=actual))
    cross_by_rank = defaultdict(list)
    for r in crossover['draft_rows']:
        cross_by_rank[r[0]].append(r)
    ctx_by_rank = {r['rank']: r for r in context}
    counts_by_rank = {r['rank']: r for r in counts}
    assert len(profiles) == len(ctx_by_rank) == len(counts_by_rank) == 16
    active_waves = {w['wave'] for p in context for w in p['windows'] if w['active']}
    assert len(active_waves) == 227
    body_rows = []
    geometry = defaultdict(list)
    per_wave_layer = defaultdict(list)
    all_rows_by_wave = defaultdict(list)
    m_counts = Counter()
    cp_flags = Counter()
    source_dbs = []
    for p in profiles:
        rank = p['rank']
        ctx = sorted(ctx_by_rank[rank]['windows'], key=lambda w: w['wave'])
        calls = sorted((c for c in p['calls'] if c[0] == 'draft'), key=lambda c: c[3][0][1])
        cross = sorted(cross_by_rank[rank], key=lambda c: c[3])
        ec = counts_by_rank[rank]['draft']
        assert len(ctx) == len(calls) == len(cross) == len(ec) == 231
        assert p['database'] == ctx_by_rank[rank]['database']
        # DB digest is carried by both original independent receipts.
        assert p['upstream_sha256'] == crossover['server_artifact_sha256'][p['database']]
        source_dbs.append({k: p[k] for k in ['rank', 'database', 'upstream_sha256', 'bytes', 'mtime_ns']})
        for w, call, cr, ex in zip(ctx, calls, cross, ec):
            wave = w['wave']
            head = call[3][0]
            first = w['first_HcPre_ns']
            assert cr[3] == w['first_HcPre']
            assert first < head[1] < call[3][-1][2]
            assert call[3][-1][2] == waves[wave][rank]['calls']['draft']['end']
            assert ex[0] == cr[7][0][2]
            assert len(cr[7]) == len(ex[1]) == 3
            row = [wave, rank, int(wave in active_waves), int(w['active']), cr[3], first,
                   cr[4], head[0], head[1], call[2], cr[2], cr[7][0][6]]
            body_rows.append(row)
            all_rows_by_wave[wave].append(row)
            for layer, (binding, ce) in enumerate(zip(cr[7], ex[1])):
                sender, experts = ce
                assert len(experts) == 8
                assert sender == binding[7] and sum(experts) == binding[8]
                assert cr[3] < binding[3] < cr[4] < head[0]
                assert 0 <= sender <= binding[6] <= 3
                if wave in active_waves:
                    v = dict(wave=wave, rank=rank, layer=layer, M=binding[6], sender=sender,
                             receiver=binding[8], zero_experts=experts.count(0),
                             max_expert=max(experts), query_branch=cr[2][layer],
                             counts=experts, F21_task=binding[3], epoch=binding[2], bank=binding[1])
                    geometry[layer].append(v)
                    per_wave_layer[wave, layer].append(v)
            if wave in active_waves:
                m_counts[cr[7][0][6]] += 1
                cp_flags[tuple(cr[2])] += 1
    assert len(body_rows) == 3696
    assert sum(len(g) for g in geometry.values()) == 10896
    # Every active wave retains all 16 participants, including an idle DP peer.
    offsets = hist['clock_contract']['offset_upper_ns']
    wave_costs = []
    epoch_value_counts = Counter()
    for wave in sorted(active_waves):
        rows = sorted(all_rows_by_wave[wave], key=lambda r: r[1])
        assert [r[1] for r in rows] == list(range(16))
        bounds = latest_change([r[8] for r in rows], [r[5] for r in rows], offsets)
        wave_costs.append(dict(wave=wave, body_to_head_start_ms=bounds,
                               active_rank_count=sum(r[3] for r in rows)))
        for layer in range(3):
            vals = per_wave_layer[wave, layer]
            assert len(vals) == 16
            epoch_value_counts[len({v['epoch'] for v in vals})] += 1
            for dp in [0, 1]:
                assert len({v['epoch'] for v in vals if v['rank'] // 8 == dp}) == 1
            assert len({v['bank'] for v in vals}) == 1
            assert sum(v['receiver'] for v in vals) == 3 * sum(v['sender'] for v in vals)
    costs = {r['wave']: r['body_to_head_start_ms'][1] for r in wave_costs}
    endpoints = read(ENDPOINTS)['endpoint_ms']
    def price(rho, added):
        out = {}
        for metric, base in endpoints.items():
            gains = []; counts_per_request = []
            for c in schedule['client']:
                deadline = c['end_ns'] if metric == 'makespan' else c['nonempty_chunk_ns'][0]
                chosen = [w for w in costs if min(r['entry'] for r in waves[w].values() if 'entry' in r) <= deadline]
                gains.append(sum(rho * costs[w] - added for w in chosen))
                counts_per_request.append(len(chosen))
            saved = max(base) - max(b - g for b, g in zip(base, gains))
            out[metric] = dict(signed_saved_ms=saved, signed_percent=100 * saved / max(base),
                               signed_per_request_ms=gains, eligible_waves_per_request=counts_per_request)
        return out
    paid = [dict(removable_exposed_fraction=rho, added_ms_per_wave=h, result=price(rho, h))
            for rho in [0, .25, .5, .75, 1] for h in [.05, .1, .25, .5]]
    thresholds = []
    for h in [.05, .1, .25, .5]:
        for metric in endpoints:
            lo = 0.; hi = 1.
            if price(hi, h)[metric]['signed_percent'] < 1:
                required = None
            else:
                for _ in range(45):
                    mid = (lo + hi) / 2
                    if price(mid, h)[metric]['signed_percent'] >= 1: hi = mid
                    else: lo = mid
                required = hi
            thresholds.append(dict(added_ms_per_wave=h, metric=metric,
                                    removable_exposed_fraction_for_historical_1pct_priority=required))
    geometries = []
    for layer, vals in sorted(geometry.items()):
        geometries.append(dict(layer=layer, rank_layer_windows=len(vals),
            sender_rows=stats([v['sender'] for v in vals]), receiver_rows=stats([v['receiver'] for v in vals]),
            zero_experts=stats([v['zero_experts'] for v in vals]),
            max_single_expert_rows=stats([v['max_expert'] for v in vals]),
            zero_sender_windows=sum(v['sender'] == 0 for v in vals),
            zero_sender_with_receivers=sum(v['sender'] == 0 and v['receiver'] > 0 for v in vals),
            no_query_branch_with_receivers=sum(v['query_branch'] == 0 and v['receiver'] > 0 for v in vals),
            all_zero_experts=sum(v['receiver'] == 0 for v in vals),
            examples_zero_sender_receives=[v for v in vals if v['sender'] == 0 and v['receiver'] > 0][:3]))
    out = dict(status='PARTIAL_THREE_BODY_SOURCE_PRICE_AND_EXPERT_GEOMETRY_REVIEW',
        precision='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL', whole_chain_complete=False,
        access_receipt='dspark-body-access.json', current_runtime_identity_revalidated=False,
        source_qualification='Local hashes match previously attested frozen source. Current171 authentication failed; no current live identity or runtime result is asserted.',
        current_C8=dict(rank_windows=3696, active_wave_rank_windows=3632, active_rank_windows=3624,
            active_waves=227, active_rank_layer_expert_windows=10896, original_all_expert_instances=11088,
            start='First actual draft HcPre TASK_ROW/start', end='Base vocabulary GEMM start after three bodies, collapse/norm/gather/selection',
            excluded='Input/context before first HcPre, vocabulary GEMM and five serial Markov score steps.',
            all_rank_body_duration_ms=stats([(r[8]-r[5])/1e6 for r in body_rows]),
            wave_frontier_ms=stats(list(costs.values())), wave_costs=wave_costs,
            favorable_all_free=price(1, 0), paid_sensitivities=paid, thresholds=thresholds,
            local_M_active_wave_windows=dict(m_counts), query_branch_flags={str(k):v for k,v in cp_flags.items()},
            pricing_contract='Original observer-containing C8; all16 EP participants retained. Saved rank-clock frontier bounds; optimistically erase each whole latest-HcPre to latest-base-head-start span and propagate every eligible preceding wave. Necessary arithmetic, communication, new synchronization, resource contention and changed arrivals are relaxed in gross. Signed rho/h figures are conditional sensitivities, not expected speedup, measured net savings or universal hardware bounds. Parents overlap existing host/context/score budgets and are not additive.',
            six_ordinary='No matching whole-body original profile binding yet; no extrapolation from small C8 M or a different residual cohort.'),
        expert_geometry=geometries,
        observer_epoch_distinct_values_per_native_wave_layer=dict(epoch_value_counts),
        expert_contract='All11088 crossover bindings join to original expert counts by rank/order/first epoch, sender counts and receiver sum; active227 native ordinal waves retain10896 expert instances. Each native wave/layer has consistent observed epoch within each DP8 group and one bank across16, while the2 DP epoch values differ in all681 groups. Observer epochs are NOT a global EP16 key. Total receivers equals top3 times senders under the original native ordinal grouping. Source-to-native association inherits the pinned observer, one ordered F21 per layer and96 triples without independent adjacent target anchors; it is not a new physical cross-rank trace or protocol proof. Zero local query/sender is not zero EP responsibility. Counts do not price elapsed work. GMM currentM guards already omit empty group arithmetic; all cross-core/rank flags and post-unpermute arena lifetime still matter.',
        body_row_columns=['wave','rank','wave_active','rank_active','first_HcPre_TASK_ROW','first_HcPre_ns',
                          'last_HcPost_TASK_ROW','base_head_TASK_ROW','base_head_start_ns','head_M','MLA_flags3','local_SP_M'],
        body_rows=body_rows, source_dbs=source_dbs, crossover_qualified_pins=crossover_pins,
        inputs=[dict(path=str(p),sha256=digest(p)) for p in [
            OUT/'embedding-context-execution-identity.json',OUT/'context-parent-receipt.json',Path(ctx_receipt['raw_path']),
            Path(RECEIPT),Path(ENDPOINTS),crossover_path,counts_path,FULL/'F21-EMPTY-EXPERT-PUBLICATION.json',
            Path(__file__)]],
        new_NPU_work=0, service_requests=0, new_profile_collection=0, remote_writes=[])
    (OUT/'dspark-three-body-frontier.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(gross=out['current_C8']['favorable_all_free'], geometry=geometries,
                         M=dict(m_counts), query_flags=out['current_C8']['query_branch_flags'],
                         frontier=out['current_C8']['wave_frontier_ms'], thresholds=thresholds),ensure_ascii=False))


if __name__ == '__main__':
    main()
