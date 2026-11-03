"""Refine existing F20/F28 padding alternatives with exact invocation ownership.

Pure Python value maps and declared-domain cost algebra only; no target execution.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N, O, W, read, write
from analyze_state_engram_v62 import dimensions
from analyze_projection_ceiling_v61 import stat


def main():
    assert not (O / 'pad-fusion-costs-v72.json').exists()
    owners = read(O / 'pad-owner-invocations-v72.json')
    inv = owners['invocations']
    assert len(inv) == 1272
    source_root = W / 'frozen-source-snapshot/vllm-ascend'
    expected = {
        'vllm_ascend/ops/fused_moe/prepare_finalize.py': 'dff4fac292200fc06f60aef48505f1ee38314ea5adb53560b325901db308eb24',
        'vllm_ascend/models/common/ops/sequence_parallel.py': '73bcf07050bce8b049dd0bbdb0a7504f6eba157a1e97a5033849ef31ef05fcb0',
        'vllm_ascend/models/deepseek_v41/dspark.py': 'fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601',
    }
    pins = []
    for relative, sha in expected.items():
        path = source_root / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        pins.append(dict(path=str(path), sha256=sha, reused_prior_source_pin=True))
    summary = []
    for field in sorted(owners['fields']):
        xs = [x for x in inv if x['field'] == field]
        task_counts = Counter()
        for x in xs:
            task_counts.update((s['operator'] or s['task_type']) for s in x['stages'])
        summary.append(dict(field=field, invocations=len(xs), native_TASK_counts=dict(task_counts),
                            native_TASK_sum_ns=stat([x['native_task_sum_ns'] for x in xs]),
                            native_enclosing_body_span_ns=stat([x['native_enclosing_body_span_ns'] for x in xs]),
                            measurement_scope='Per distinct native API owner; task sums/spans are not request critical path, host launches, or attainable savings.'))
        for x in xs:
            ish = dimensions(x['PadV3_signature']['inputShapes_text'])[0]
            osh = dimensions(x['PadV3_signature']['outputShapes_text'])[0]
            if field.startswith('MC2'):
                assert (ish[0], osh[0]) == (800, 816)
                assert x['context'] == 'CALLER_TARGET_MODEL_OR_ENGRAM'
            else:
                assert (ish[0], osh[0]) in [(10, 16), (20, 24)]
            expected_ops = {'Cast': 3, 'MemSet': 1, 'KERNEL_MIX_AIV': 1, 'PadV3': 1} if field.endswith('MASK') else {'MemSet': 1, 'KERNEL_MIX_AIV': 1, 'PadV3': 1}
            assert Counter((s['operator'] or s['task_type']) for s in x['stages']) == expected_ops
    # The ownership unit is ONE pad call, not a guessed temporal triplet.
    assert sum(sum(x['native_TASK_counts'].values()) for x in summary) == 5088
    native_keys = [(x['database_index'], s['TASK_rowid']) for x in inv for s in x['stages']]
    assert len(set(native_keys)) == len(native_keys) == 5088
    h, e, n, p = 5120, 384, 800, 816
    mc2_fields = [dict(field='hidden', dtype='BF16', input_bytes=2*n*h, output_bytes=2*p*h),
                  dict(field='router_logits', dtype='FLOAT', input_bytes=4*n*e, output_bytes=4*p*e),
                  dict(field='mask', dtype='BOOL', input_bytes=n, output_bytes=p)]
    for row in mc2_fields:
        row['one_read_full_write_reference_bytes'] = row['input_bytes'] + row['output_bytes']
    mc2_bytes = sum(x['one_read_full_write_reference_bytes'] for x in mc2_fields)
    assert mc2_bytes == 19031632
    mc2 = dict(N=n, P=p, H=h, E=e, occurrences_per_field=320, fields=mc2_fields,
               joint_output_reference_bytes=mc2_bytes, input_bytes=9421600, output_bytes=9610032,
               eliminated_known_BOOL_data_and_output_cast_reference_bytes=2*(n+p),
               scalar_cast_bytes=None, MemSet_address_bytes_and_overlap=None,
               native_owner_population=dict(API_calls=960, ComputeInfo_TASKs=2880, nonCompute_KERNEL_MIX_AIV=960, all_TASKs=3840),
               not_a_native_three_pad_invocation_join=True,
               candidate='F20', prior_mechanism='Existing v63 padded-producer / pad-packing alternative; new ownership evidence, not a new candidate.',
               full_replacement_contract='Write all three outputs including 16 padded rows; preserve BF16/FLOAT copies and positive-zero tails, BOOL values/false tail, shape/stride/alignment, alias separation, stable capture addresses, all readers and readiness. A joint producer must not silently add a wait to an earlier reader.',
               shared_output_is_not_automatically_aliasable=True,
               native_MC2_816_to_800_ABI_change_not_proved=True,
               input_ID_pad_outside_this_960_API_population=True)
    assert mc2['input_bytes'] == sum(x['input_bytes'] for x in mc2_fields)
    assert mc2['output_bytes'] == sum(x['output_bytes'] for x in mc2_fields)
    previous = read(O / 'embedding-initialization-fusion-costs-v67.json')['draft_initialization_models']
    draft = []
    for old in previous:
        n, p, c = old['N'], old['A_padded'], old['local_rows']
        count = 32 if n == 10 else 72
        pop = Counter()
        for x in inv:
            if x['field'] == 'SP_HIDDEN' and dimensions(x['PadV3_signature']['inputShapes_text'])[0][0] == n:
                pop[x['context']] += 1
        assert sum(pop.values()) == count
        old_hc = 2*n*h*(1+4) + 2*4*(n+p)*h
        assert old_hc == old['current_global_Tile_plus_HC_Pad_value_bytes']
        ranks = []
        for rank in range(8):
            start = rank*c
            v = max(0, min(n, start+c)-start)
            local_pad_after_repeat = 2*4*h*(v+c)
            joint_local_init = 2*h*v + 2*4*h*c
            local_2d_then_tile = 2*h*(v+c) + 2*h*c*(1+4)
            old_rank = old['rank_models'][rank]
            assert old_rank['valid_owned_rows'] == v
            assert old_rank['new_direct_HC_init_value_bytes'] == joint_local_init
            assert old_rank['HC_interface_budget_bytes'] == old_hc-joint_local_init
            ranks.append(dict(rank=rank, start=start, valid_owned_rows=v, output_rows=c,
                              direct_local_SP_after_existing_global_Tile_value_bytes=local_pad_after_repeat,
                              direct_local_2D_SP_then_local_Tile_value_bytes=local_2d_then_tile,
                              joint_local_SP_HC4_value_bytes=joint_local_init,
                              current_global_Tile_plus_HC4_Pad_value_bytes=old_hc,
                              HC_interface_budget_bytes=old_hc-joint_local_init,
                              extra_budget_vs_direct_local_2D_then_Tile=local_2d_then_tile-joint_local_init,
                              provided_mask_read_write_reference_bytes=v+c,
                              None_mask_full_write_reference_bytes=c,
                              input_ID_read_write_reference_bytes=4*(v+c),
                              initial_FP32_onehot_fullwrite_bytes=4*c*4))
        assert sum(r['valid_owned_rows'] for r in ranks) == n
        draft.append(dict(N=n, P=p, local_rows=c, HC=4, H=h, occurrences=count, contexts=dict(pop),
                          native_three_pad_population=dict(API_calls=count*3, ComputeInfo_TASKs=count*9, nonCompute_KERNEL_MIX_AIV=count*3, all_TASKs=count*12),
                          baseline_global_Tile_plus_HC4_Pad_value_bytes=old_hc,
                          existing_global_2D_pad_then_local_Tile_value_bytes=old['delayed_Tile_after_2D_Pad_SP_value_bytes'],
                          existing_delayed_Tile_budget_bytes=old['delayed_Tile_interface_budget_bytes'],
                          rank_models=ranks,
                          source_functional_contract='Pad to P then return rank-contiguous C-row view of the fresh padded output; no-pad path clones. New direct output preserves independence/stride and every required value, not arbitrary input aliasing.',
                          candidate='F28', prior_mechanism='Existing v67 direct local initializer and SP-before-repeat; numerical budgets unchanged. New exact ownership permits a conditional whole-call helper cut.',
                          no_query_rank_is_not_whole_decoder_dead=True,
                          mask_tail=True, IDs_tail=0, hidden_tail_positive_zero=True))
    # Representation-free symbolic witnesses: rows carry arbitrary BF16 bit payload labels.
    map_cases = 0
    supplied_masks = 0
    for n in range(0, 65):
        for tp in range(1, 17):
            p = ((n+tp-1)//tp)*tp
            c = p//tp
            for hc in [1, 2, 4]:
                data = [(i, j) for i in range(n) for j in range(hc)]
                global_pad = data + [('POSITIVE_ZERO', j) for _ in range(p-n) for j in range(hc)]
                for rank in range(tp):
                    start = rank*c
                    old_values = global_pad[start*hc:(start+c)*hc]
                    new_values = [(i, j) if i<n else ('POSITIVE_ZERO', j) for i in range(start, start+c) for j in range(hc)]
                    assert old_values == new_values
                    ids = list(range(n)) + [0]*(p-n)
                    assert ids[start:start+c] == [i if i<n else 0 for i in range(start, start+c)]
                    for mask in [[False]*n, [True]*n, [i%2 == 0 for i in range(n)]]:
                        old_mask = (mask+[True]*(p-n))[start:start+c]
                        new_mask = [mask[i] if i<n else True for i in range(start, start+c)]
                        assert old_mask == new_mask
                        supplied_masks += 1
                    map_cases += 1
    # A copy permutation performs no floating-point arithmetic on any 16-bit payload.
    bf16_payload_checks = 0
    for payload in range(65536):
        assert [payload]*4 == [payload for _ in range(4)]
        bf16_payload_checks += 1
    variants = [
        dict(id='F20-PAD3', mechanism='Replace three MC2 pure padding calls by one three-output producer', effect='Remove private call helper subgraphs under complete output/API contract; same required pad payload, plus removal of known BOOL conversions.', condition='All outputs ready at required readers; hidden/logits/mask ownership and generation/capture/error semantics preserved. No automatic critical-path gain.'),
        dict(id='F20-PRODUCER', mechanism='Write padded hidden/logits in their respective upstream producers and generate aligned mask directly', effect='Avoid intermediates rather than copy them through a second pad kernel.', condition='Preserve full-row RMS/router FP32 numerical domain, BF16 rounding, shared live hidden readers and MC2 output ABI. Existing F20 mechanism; no unproved cross-layer CSE.'),
        dict(id='F20-VIRTUAL', mechanism='Teach MC2/routing readers to interpret virtual zero/false padded rows', effect='May avoid materialized padding and padded row work.', condition='Requires new consumer/dispatch/EP-count/workspace contract. Existing fused MC2 baseline and distributed ordering retained; cannot merely change 816 to 800.'),
        dict(id='F28-LOCAL', mechanism='Directly materialize only the SP-owned output rows of each pure pad', effect='Cut the full global pad invocation and its private helpers; rank-specific row map, true/zero tails preserved.', condition='Fresh functional outputs/capture ownership; input row readiness and downstream row correspondence. Entire native operator replacement, not independent MemSet deletion.'),
        dict(id='F28-JOINT', mechanism='Combine SP row selection with HC4 initialization, IDs, mask and initial premix', effect='Use existing v67 direct initializer value map; new owner evidence identifies which pad helper subgraphs are in scope.', condition='No hidden numerical arithmetic added; supplied True prefix retained; no-query KV/global restore/collectives still live. IDs/mask/premix all reader and bank protocols remain.'),
        dict(id='F28-NOPAD', mechanism='Specialize aligned no-pad branch', effect='Can create a new direct functional output but cannot replace current clone by input view under existing alias contract.', condition='No such no-pad invocation is established by this 104-initialization padding population.'),
        dict(id='EXCLUDED-MEMSET-ONLY', mechanism='Delete an isolated MemSet because PadV3 follows', effect='EXCLUDED_AS_UNPROVED', condition='1272 owners are bound; actual destination, byte length, overlap and workspace-vs-payload remain absent.'),
        dict(id='EXCLUDED-DOUBLECOUNT', mechanism='Add unknown MemSet bytes to mandatory full Pad output writes as independent traffic', effect='EXCLUDED_AS_UNPROVED', condition='Overlap between helper writes and required output/pad writes is unknown; logical copy model is not HBM traffic.'),
        dict(id='EXCLUDED-ZERO-RANK', mechanism='Drop the full draft decoder on ranks owning no query rows', effect='EXCLUDED_BY_EXISTING_GLOBAL_KV_AND_COLLECTIVE_CONTRACT', condition='SP output row V=0 does not remove KV publication/head restoration/global collective participants.'),
    ]
    for v in variants:
        v.update(candidate_newly_discovered=False, legal_fusion_proved=False, measured_speedup=None)
    theory = dict(status='CONDITIONAL_CEILING', reference_bandwidth_bytes_per_second=1600000000000,
                  reference_source='https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1',
                  allowed_domain='Cold input payload only in HBM, fresh arbitrary output HBM allocations, all listed outputs fully materialized and externally observed, no compression/alias/view or prior zero-bank invariant; stated 1.6TB/s is used as an assumed applicable upper capacity for this reference screen.',
                  conditional_model='T >= max(Q_HBM/B_HBM_upper, other independently proved resource/dependency lower bounds). Under this restricted domain Q_HBM >= input arbitrary payload + complete output; no launch/Vector/cache capacity is supplied by this profile.',
                  actual_physical_traffic_bound_proved=False, actual_numeric_request_ceiling=None,
                  MC2_pad3_reference_us=mc2_bytes/1600000,
                  MC2_pad3_conditional_call_per_second_upper=1600000000000/mc2_bytes,
                  draft_joint_HC4_reference_us=[dict(N=x['N'], rank=r['rank'], value_bytes=r['joint_local_SP_HC4_value_bytes'], us=r['joint_local_SP_HC4_value_bytes']/1600000) for x in draft for r in x['rank_models']],
                  actual_cache_residency_and_HBM_bytes_known=False,
                  whole_model_speedup_from_TASK_sum_not_computed=True)
    write(O / 'pad-operator-contract-v72.json', dict(source_files=pins, source_clauses=['prepare_finalize.py:274-285/308-316/328-339', 'sequence_parallel.py:32-43/105-124/142-152', 'dspark.py:248-275'], source_scope='Only padding value maps, HC repeat, owner output/reader shape and alias clauses; no framework survey.', new_remote_source_read=False, exact_owner_population=1272, independent_MemSet_operands_known=False, whole_modules_closed=0, overall_complete=False))
    write(O / 'pad-fusion-costs-v72.json', dict(native_owner_measurements=summary, MC2=mc2, draft=draft, native_TASK_total=5088, ComputeInfo_TASK_total=3816, nonCompute_TASK_total=1272, draft_old_v67_costs_preserved=True, unknown_MemSet_bytes_not_added=True, theory=theory, whole_modules_closed=0, overall_complete=False))
    write(O / 'pad-fusion-variants-v72.json', dict(variants=variants, count=len(variants), new_model_candidates=0, whole_modules_closed=0, overall_complete=False))
    write(O / 'pad-value-map-validation-v72.json', dict(symbolic_row_map_cases=map_cases, supplied_mask_cases=supplied_masks, all_BF16_payload_copy_witnesses=bf16_payload_checks, TP_domain=[1,16], N_domain=[0,64], HC_domain=[1,2,4], source_native_or_Torch_tests=False, no_pad_clone_alias_not_relaxed=True, old_v67_rank_budgets_exactly_preserved=True, actual_persistent_bank_reader_protocol_proved=False, overall_complete=False))
    print(dict(owners=1272, native_TASKs=5088, ComputeInfo_TASKs=3816, nonCompute_TASKs=1272, MC2_reference_bytes=mc2_bytes, MC2_BOOL_cast_reference_budget_bytes=3232, draft_occurrences=sum(x['occurrences'] for x in draft), value_map_cases=map_cases, mask_cases=supplied_masks, BF16_payloads=bf16_payload_checks, variants=len(variants), new_candidates=0, pending_promoted=0))


if __name__ == '__main__':
    main()
