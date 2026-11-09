#!/usr/bin/env python3
"""Offline worker/capture branch review and original-C8 family binding.

Derives shape/branch requirements, not native graph or numerical admission.
"""
import ast
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import zlib

sys.dont_write_bytecode = True
BASE = Path('/Users/wio/work/branch-fusion-audit-20261104')
FROZEN = Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/frozen-source-snapshot')
OUT = Path('audit/execution-search')
sys.path.insert(0, str(BASE / 'tools'))
from price_vocab_selection_fusion import records
from price_attention_send_epilogue import ENDPOINTS


def read(p):
    return json.loads(Path(p).read_bytes())


def digest(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def main():
    previous = read(OUT / 'dspark-body-publication.json')
    body_path = OUT / 'dspark-three-body-frontier.json'
    assert digest(body_path) == previous['full_local_artifact']['sha256']
    body = read(body_path)
    waves, _, schedule, _, _ = records()
    cross_path = BASE / 'full-review/F21-SAME-COHORT-CROSSOVER.json'
    cross = read(cross_path)
    expected = next(p['sha256'] for p in body['inputs'] if p['path'] == str(cross_path))
    assert digest(cross_path) == expected
    cross_by_first = {(r[0], r[3]): r for r in cross['draft_rows']}
    branch_hist = Counter(); observed = defaultdict(Counter); companions = []
    rows_by_wave = defaultdict(list)
    for r in body['body_rows']:
        if r[2]: rows_by_wave[r[0]].append(r)
    families = []
    for bmax in range(1, 5):
        n = 5 * bmax; m = (n + 7) // 8
        for b in range(1, bmax + 1):
            q = [max(0, min(m, 5*b - tp*m)) for tp in range(8)]
            assert sum(q) == 5*b
            families.append(dict(entry='real_metadata', own_B=b, max_B=bmax,
                model_input_rows=n, local_SP_rows=m, original_head_rows=5*b, CP_actual_Q=q,
                MC2_sender_rows=q, metadata='Present; Q0 still executes CP output restore/O projection',
                native_admission=False))
        families.append(dict(entry='dummy_no_metadata', synthetic_B=1, max_B=bmax,
            model_input_rows=n, local_SP_rows=m, original_head_rows=5, CP_actual_Q=[0]*8,
            MC2_sender_rows=[max(0,min(m,n-tp*m)) for tp in range(8)],
            metadata='None; attention immediately zeroes its output; all3 EP calls remain',
            native_admission=False))
    assert len(families) == 14
    for wave, rs in sorted(rows_by_wave.items()):
        own = {dp: len(waves[wave][8*dp].get('scheduled', {})) for dp in [0,1]}
        bmax = max(own.values()); assert 1 <= bmax <= 4
        n = bmax * 5; m = (n + 7) // 8
        for r in rs:
            rank = r[1]; dp, tp = divmod(rank,8); b = own[dp]
            active = r[3]; assert bool(b) == bool(active)
            assert r[9] == 5*(b if active else 1) and r[11] == m
            actual_q = max(0,min(m,5*b-tp*m)) if active else 0
            assert r[10] == [int(actual_q > 0)] * 3
            valid = 5*b if active else n
            expected_sender = max(0,min(m,valid-tp*m))
            cr = cross_by_first[rank, r[4]]
            assert all(x[7] == expected_sender for x in cr[7])
            branch = 'real_Q_positive' if actual_q else ('real_Q0_with_metadata' if active else 'dummy_metadata_None')
            branch_hist[branch] += 1
            observed[(b if active else 'dummy',bmax)][branch] += 1
            if not active: companions.append(dict(wave=wave,rank=rank,synthetic_B=1,max_B=bmax,
                expected_sender=expected_sender,observed_sender=[x[7] for x in cr[7]],
                observed_receiver=[x[8] for x in cr[7]],first_HcPre_TASK=r[4],head_TASK=r[7]))
    assert sum(branch_hist.values()) == 3632
    assert len(companions) == 8
    # Intentionally more generous than deleting only the dummy side: erase the
    # entire already priced 16-rank parent on every wave with any idle DP.
    idle_waves = sorted({r['wave'] for r in companions})
    wave_costs = {r['wave']:r['body_to_head_start_ms'][1] for r in body['current_C8']['wave_costs']}
    endpoints = read(ENDPOINTS)['endpoint_ms']; idle_price = {}
    for metric, base in endpoints.items():
        gains = []
        for c in schedule['client']:
            deadline = c['end_ns'] if metric == 'makespan' else c['nonempty_chunk_ns'][0]
            eligible = [w for w in idle_waves if min(r['entry'] for r in waves[w].values() if 'entry' in r) <= deadline]
            gains.append(sum(wave_costs[w] for w in eligible))
        saved = max(base)-max(b-g for b,g in zip(base,gains))
        idle_price[metric] = dict(signed_saved_ms=saved, signed_percent=saved/max(base)*100,
                                  signed_per_request_ms=gains)
    config_path = FROZEN / 'vllm-ascend/vllm_ascend/ascend_config.py'
    text = config_path.read_text(); tree = ast.parse(text)
    guard = next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and ast.unparse(n.test) == 'self.enable_fused_mc2 == 1 and self.multistream_overlap_shared_expert')
    assert ast.unparse(guard.body[0]) == 'self.multistream_overlap_shared_expert = False'
    old_path = BASE / 'full-review/SHARED-OVERLAP-C8.json'; old = read(old_path)
    assert digest(config_path) == old['exclusion_evidence']['config-exclusion-source.json']['data']['sha256']
    assert digest(BASE/old['receipt']) == old['receipt_sha256']
    assert old['candidate_samples'] == 0 and old['service_speedup'] is None
    snap_path = BASE / 'full-review/EP0-CONFIG-SNAPSHOT.json'; snap = read(snap_path)
    assert digest(snap['source']['path']) == snap['source']['sha256']
    f = snap['fields']
    assert f['root.vllm_config.scheduler_config.max_num_seqs'] == [4]
    assert f['root.vllm_config.speculative_config.enforce_eager'] == [True]
    assert f['root.ascend_config.multistream_overlap_shared_expert'] == [False]
    assert f['root.ascend_config.finegrained_tp_config.lmhead_tensor_parallel_size'] == [0]
    out = dict(status='PARTIAL_ACTUAL_WORKER_FAMILY_AND_POLICY_REVIEW',whole_chain_complete=False,
        precision='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',current_runtime_identity_revalidated=False,
        source_review='worker-capture-source-review.json',
        correction=dict(previous_claim='Frozen shared/routed preparation and Gate-Up are already overlapped',
            disposition='WITHDRAW_UNCONDITIONAL_ENABLED_CLAIM',
            actual_frozen_source_rule=ast.unparse(guard.test)+' => '+ast.unparse(guard.body[0]),
            exact_source_sha=digest(config_path),old_candidate_samples=0,old_measured_speedup=None,
            implication='Optional implementation exists, but fused MC2 disables its flag. Flag-only recipe stopped; explicit supported co-residency/scheduling redesign still pending. Original whole-parent prices unchanged. Do not use this correction to credit a whole shared path as free or infer hardware impossibility.'),
        source_families=families,
        family_contract='Source-derived14 design combinations for maxB1..4:10 positive-own-B real metadata cases +4 synthetic-B1 dummy cases. This is not the former ownB0 shape enumeration, an exhaustive graph family or native admission. Body input5*DPmaxB differs from head5*ownB: frozen fine-grained lmhead padding helper is false (size0), while ordinary vocabulary TP remains8. Actual CP rank/Q shape, num_reqs/QSL, real vs None metadata, input/padded rows, mask and cache/event/workspace generations still enter the contract. Warmup/profile/capture and errors are separate caller modes.',
        original_C8=dict(active_wave_rank_windows=3632,observed_branches=dict(branch_hist),
            observed_family_counts=[dict(own_B=b,max_B=k,branches=dict(v)) for (b,k),v in observed.items()],
            idle_companion_windows=companions,idle_waves=idle_waves,
            more_generous_complete_EP16_body_parent_free_on_idle_waves=idle_price,
            idle_budget_contract='Erase even real peer three-body/head-input work on these waves; original clock/frontier/deadline and observer assumptions retained. This favorable source-local sensitivity bounds the narrower dummy-only recipe within the same schedule, not target40-body idle work, changed real expert service/arrival geometry, ordinary loads or hardware universally.',
            source_native_binding='All3632 branch and SP/head geometries and all3 expected sender counts match old native HcPre/head/crossover data. Inherits previous observer/rank-order/96-unanchored-triple limitations; not new cross-rank trace or numeric validation.'),
        concrete_pending=['Build real-metadata scratch-cache warmup/capture at the body cut, not current zero-attention dummy graph',
            'Effective dispatcher and synchronized all-rank capture/fallback decision; no target descriptor or kwargs-address shortcut',
            'Producer/output/workspace generation and changed-mask real receiver geometry per primitive bits',
            'Supported shared/routed core partition, actual M1/2/3 and each draft weight block bits; net service/resource cost',
            'Broader inactive-DP target40 + draft participation; six ordinary matched sender/receiver/timing and branch frequency',
            'kernel_warmup and workspace cleanup callees missing from selected frozen snapshot; preserve explicit unreviewed frontier'],
        stopped_scoped_recipes=['Only flip DSpark eager flag and replay existing no-attention dummy template as real body',
            'Use one M3 graph for real Q0 and None-metadata dummy',
            'Only flip multistream_overlap_shared_expert with unchanged fused MC2 policy',
            'Skip inactive EP rank entirely',
            'Standalone dummy-draft deletion for original-C8 TTFT1pct priority; even larger whole-parent-on-idle-wave erasure below priority'],
        inputs=[dict(path=str(p),sha256=digest(p)) for p in [body_path,OUT/'dspark-body-publication.json',cross_path,
            config_path,old_path,BASE/old['receipt'],snap_path,Path(snap['source']['path']),Path(ENDPOINTS),Path(__file__)]],
        new_NPU_work=0,service_requests=0,new_profile_collection=0,remote_writes=[])
    (OUT/'worker-capture-family-frontier.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(branches=dict(branch_hist),idle_waves=idle_waves,idle_price=idle_price,
                         policy=out['correction']['actual_frozen_source_rule'],source_families=len(families))))


if __name__ == '__main__':
    main()
