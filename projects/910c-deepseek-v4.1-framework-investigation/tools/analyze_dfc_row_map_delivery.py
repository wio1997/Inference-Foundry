#!/usr/bin/env python3
"""Check route ownership with opaque words and reconcile matched saved evidence."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import median
import json

BASE = Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT = Path('audit/execution-search')

def stats(v):
    return dict(n=len(v), min=min(v), median=median(v), max=max(v))

def main():
    pins = {}
    def read(relative):
        p = BASE / relative
        b = p.read_bytes()
        pins[str(p)] = sha256(b).hexdigest()
        return json.loads(b)

    raw = read('.cache/v168-f21-physical-ids-on106.json')
    old = read('full-review/F21-PHYSICAL-ROUTE-DELIVERY.json')
    packing = read('full-review/F21-PEER-INPUT-PACKING.json')
    value = read('full-review/F21-PEER-INPUT-VALUE.json')
    native = read('full-review/runs/f21-peer-native-3254db03f36e-result.json')
    route_cases = read('full-review/F21-PEER-INPUT-ROUTES.json')
    loop = read('full-review/F21-PEER-INPUT-LOOP.json')
    for path, digest in old['inputs_sha256'].items():
        if path in pins:
            assert pins[path] == digest
    for path, digest in value['sources'].items():
        p = BASE / path
        assert sha256(p.read_bytes()).hexdigest() == digest
        pins[str(p)] = digest
    deps = json.loads((OUT/'dfc-row-map-dependencies.json').read_text())
    current = next(r for r in deps['sources'] if r['runtime_path'].endswith('dispatch_ffn_combine_kernel.hpp'))
    assert current['runtime_sha256'] == old['owner_source']['sha256']
    profiles = {p['rank']: p['rows'] for p in raw['profiles']}
    assert sorted(profiles) == list(range(16))
    groups = []; checked_receivers = 0; word_bytes = 0; sender_dummies = 0
    # A code/scale record is opaque. Include every byte encoding and distinctive
    # trailing bytes; no floating arithmetic or quantization emulation is used.
    def record(src, row):
        return bytes((i + src*17 + row*53) & 255 for i in range(5120)) + bytes((0x80, src, row, 0x7f)) + bytes([0xA5])*28
    payloads = {(src, row): record(src, row) for src in range(16) for row in range(3)}
    for layer in [0, 20]:
        timelines = [sorted((r for r in profiles[s] if r[2] == layer), key=lambda r: r[0]) for s in range(16)]
        assert all(len(t) == 227 for t in timelines)
        for ordinal, rows in enumerate(zip(*timelines)):
            for src, r in enumerate(rows):
                assert len(r[7]) == 3 and all(len(x) == 6 for x in r[7])
                if r[5] == 0:
                    assert all(set(x) == {384} for x in r[7]); sender_dummies += 18
                else:
                    assert r[5] == 3 and all(len(set(x)) == 6 and all(0 <= e < 384 for e in x) for x in r[7])
            result = dict(layer=layer, ordinal=ordinal, native_task_rows=[r[8][0] for r in rows],
                          active_sender_rows=sum(r[5] for r in rows), receiver_counts=[r[6] for r in rows])
            receiver_metrics = []
            for dst in range(16):
                refs = [(src, t//6, e%24) for local in range(24) for src, r in enumerate(rows)
                        for t, e in enumerate(sum(r[7], [])) if e == dst*24+local]
                assert [sum(x[2] == e for x in refs) for e in range(24)] == rows[dst][6]
                assert len(refs) <= 288
                unique = len({(s, row) for s, row, _ in refs})
                old_copy_issues = sum((sum(x[0] == src and x[2] == e for x in refs)+1)//2 for e in range(24) for src in range(16))
                first = next((e for e in range(24) if rows[dst][6][e]), None)
                first_unique = len({(s, row) for s, row, e in refs if e == first}) if first is not None else 0
                receiver_metrics.append(dict(rows=len(refs), unique_source_rows=unique, old_expert_pair_copy_issues=old_copy_issues,
                                             first_expert_unique_rows=first_unique, extra_rows_for_full_slab=unique-first_unique))
            # Given the ORIGINAL permutation R, U[R[t]]=floor(t/6) preserves
            # every record, regardless of tie order. Three within-expert orders
            # exercise this address identity; none claims actual Sort stability.
            for tie in [0, 1, 2]:
                sender = []
                for src, r in enumerate(rows):
                    ids = sum(r[7], [])
                    order = sorted(range(18), key=lambda t: (ids[t], t if tie == 0 else -t if tie == 1 else (7*t)%19))
                    inverse = [0]*18
                    for d, t in enumerate(order): inverse[t] = d
                    u = [None]*18
                    for t, d in enumerate(inverse): u[d] = t//6
                    expanded = [payloads[src, t//6] for t in order]
                    assert expanded == [payloads[src, u[d]] for d in range(18)]
                    sender.append((ids, order, u, expanded))
                for dst in range(16):
                    expected = []; gathered = []
                    for expert in range(24):
                        for src, (ids, order, u, expanded) in enumerate(sender):
                            for d, t in enumerate(order):
                                if ids[t] == dst*24+expert:
                                    expected.append(expanded[d]); gathered.append(payloads[src, u[d]])
                    assert b''.join(expected) == b''.join(gathered)
                    assert len(expected) == sum(rows[dst][6]); checked_receivers += 1
                    word_bytes += len(expected)*5152
            result['receiver_metrics'] = receiver_metrics
            result['maximum_rows'] = max(r['rows'] for r in receiver_metrics)
            result['maximum_unique_rows'] = max(r['unique_source_rows'] for r in receiver_metrics)
            groups.append(result)

    # Recompute the archived paired statistic, not just trust a report label.
    workers = native['ranks']; assert sorted(r['rank'] for r in workers) == list(range(16))
    legacy = []; native_inverse_cases = []
    for case in route_cases['cases']:
        cid = case['id']; samples = {}
        for arm in ['control-before', 'candidate', 'control-after']:
            vals = [w['arms'][arm]['cases'][cid] for w in workers]
            for w, v in zip(workers, vals):
                before = w['arms']['control-before']['cases'][cid]
                assert v['exact'] == before['exact'] and v['input_hashes'] == before['input_hashes']
                assert len(v['exact']['outputs']) == len(v['input_hashes']) == 8
                assert len(v['rounds_us_per_call']) == 15
                if arm == 'candidate':
                    assert v['packets']['all_5152_bytes_equal_per_token']
                    assert v['packets']['observed_tiling'] == [21000,0,18,1,1]
            for mode in (['0','1','2'] if arm == 'candidate' else ['0']):
                samples[arm, mode] = [max(v['rounds_us_per_call'][j][mode] for v in vals) for j in range(15)]
        changes = {name: [a-b for a,b in zip(samples['candidate',before], samples['candidate',after])]
                   for name, before, after in [('free_preposition','0','1'),('free_preposition_and_skip_quant','0','2'),('skip_quant_incremental','1','2')]}
        for w in workers:
            src = w['rank']; v = w['arms']['candidate']['cases'][cid]
            inverse = v['first_witness']['inverse']; assert sorted(inverse) == list(range(18))
            route = case['route'][src]; ids = sum(route['ids'], [])
            ids = [e if route['valid'][t//6] else 384 for t,e in enumerate(ids)]
            order = sorted(range(18), key=lambda t: inverse[t])
            assert [ids[t] for t in order] == sorted(ids)
            u = [None]*18
            for t, d in enumerate(inverse):u[d] = t//6
            original = [payloads[src,t//6] for t in order]
            restored = [payloads[src,u[d]] for d in range(18)]
            assert b''.join(original) == b''.join(restored)
            wrong = [payloads[src,d//6] for d in range(18)]
            native_inverse_cases.append(dict(case=cid,rank=src,original_inverse=inverse,derived_source_rows=u,
                                             opaque_payload_equal=True,sorted_position_div_topK_is_wrong=original!=wrong))
        legacy.append(dict(id=cid, paired_samples_us=changes, paired_saved_us={k:stats(v) for k,v in changes.items()},
                           all8_final_output_hashes_equal_in_saved_receipt=True, packet_scope='Only positive slots, synthetic inputs/weights; archived diagnostic21000, not new implementation/allmodel/primitives qualification.'))
    # Separate all-route geometric opportunity from five-route native timing.
    result = dict(status='BOUNDED_ROUTE_MAP_CPU_AND_MATCHED_ARCHIVED_INPUT_VALUE',
                  current_owner_source_sha256=current['runtime_sha256'], route_groups=len(groups), original_rank_layer_calls=7264,
                  receiver_byte_comparisons=checked_receivers, opaque_record_bytes_compared=word_bytes,
                  dummy_expanded_records_covered=sender_dummies, byte_mismatches=0,
                  by_layer=[dict(layer=l, groups=227, maximum_receiver_rows=stats([g['maximum_rows'] for g in groups if g['layer']==l]),
                                 maximum_receiver_unique_rows=stats([g['maximum_unique_rows'] for g in groups if g['layer']==l]),
                                 receiver_extra_placements=stats([m['rows']-m['unique_source_rows'] for g in groups if g['layer']==l for m in g['receiver_metrics']]),
                                 full_slab_requires_more_than_first=sum(m['extra_rows_for_full_slab']>0 for g in groups if g['layer']==l for m in g['receiver_metrics'])) for l in [0,20]],
                  groups=groups, legacy_five_routes=legacy, legacy_native_inverse_cases=native_inverse_cases, legacy_diagnostic_abi=loop['diagnostic_abi'],
                  old_reference_us_per_call=packing['target_net_us_per_call_reference'],
                  scope='Actual M3/K6/E24/EP16 target layers0/20 only,227capturedcalls/layer. Original Sort permutation not archived; CPU treats it as supplied and varies ties, proving finite opaque byte/address identity only. Native five-route free-delivery counterfactual keeps alternate receiverlayout, originalsort/counts/flags/epilogue and syntheticinputs, not a universalphysical/request bound.',
                  findings=['Keep original inverseR through finalUnpermute. Derive newU[R[t]]=t//6 without replacing originaltieorder. Dummy384 remainsinoriginalR/counts/record domain but contributes no realreceiver expert.',
                            'Deleting receivergmA writes doesnot automatically freeitsarena: ptrA aliases ptrPermutedToken for SwiGLU activation. offsetD and inverseR remain live through Unpermute after finalCrossRankSync.',
                            'Retaining originalcountpublication and allpergroupflags gives a conservativecandidate boundary. Newpeerrow/gather-to-L1 readers delay sourcebankretirement until actualremoteGMM completion; sourcecopydone insufficient.',
                            'Five oldfixedconsumer input-only recipes remain stopped underoldlocalbudget: evenfreeprepositioning median1.543–6.922us vs14.2497reference. Synthetic/alternate-layout sensitivity, not measured newplan or hardfullparent bound. ChangedGMM/earlierpublication/largeM/crosslayer parent remainspending.'],
                  pending=['Native mapscatter/publish/addressloader and perprimitivebits including inactive/dummy/padding; not only finaloutputhashes.',
                           'Actual earliest-reader cost, pergroup wait, same-generation sourcebank lastreader/capture/resource; unique volume alone doesnotpredict speed.',
                           'Other38targetlayers, drafttop3/localE8 and sixordinary matchedroutes not covered bytheseIDs; sourcelegalpartial masks additionalgap.',
                           'BroaderGMM/sharedschedule or changedpublication has independent paidbudget. No re-run of originalfixedconsumer input-only variants.'],
                  inputs=[dict(path=p,sha256=h) for p,h in pins.items()],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  whole_chain_complete=False,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
    (OUT/'dfc-row-map-delivery.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['route_groups','original_rank_layer_calls','receiver_byte_comparisons','opaque_record_bytes_compared','dummy_expanded_records_covered','byte_mismatches','by_layer']},ensure_ascii=False))
    print(json.dumps([dict(id=r['id'],paired_saved_us=r['paired_saved_us']) for r in legacy],ensure_ascii=False))

if __name__ == '__main__':
    main()
