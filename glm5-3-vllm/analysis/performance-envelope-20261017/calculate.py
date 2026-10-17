#!/usr/bin/env python3
"""Reproduce conditional resource roofs. No model/server access; stdlib only.

This is NOT an attainable latency predictor or a universal algorithmic bound.
All linear matrices are relaxed to one byte/weight and the fastest INT8 rate,
including real FLOAT exceptions. Their omission makes this more optimistic.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'inputs.json').read_text())
C = DATA['model_config']
L = C['num_hidden_layers']
D = C['hidden_size']
H = C['num_attention_heads']
E = C['n_routed_experts']
K = C['num_experts_per_tok']
ND = C['first_k_dense_replace']
NM = L - ND
NI = C['indexer_types'].count('full')
RANKS = DATA['dies_per_stage']
BW = RANKS * DATA['hbm_bytes_s_per_die_reference']
KV = DATA['kv_layout_snapshot']['33']['representation_bytes_per_logical_token_per_rank']


def ledger():
    a = {
        'q_a': D * C['q_lora_rank'],
        'q_b': C['q_lora_rank'] * H * C['qk_head_dim'],
        'kv_a': D * (C['kv_lora_rank'] + C['qk_rope_head_dim']),
        'kv_b': C['kv_lora_rank'] * H * (C['qk_nope_head_dim'] + C['v_head_dim']),
        'o': H * C['v_head_dim'] * D,
    }
    expert = 3 * D * C['moe_intermediate_size']
    index = (C['q_lora_rank'] * C['index_n_heads'] * C['index_head_dim']
             + D * C['index_head_dim'] + D * C['index_n_heads'])
    fixed = (L * sum(a.values()) + ND * 3 * D * C['intermediate_size']
             + NM * (C['n_shared_experts'] * expert + D * E) + NI * index)
    return {'attention_per_layer': a, 'one_expert': expert,
            'index_per_full_layer': index, 'fixed_body': fixed,
            'routed_per_selected_expert_across_layers': NM * expert,
            'lm_head': D * C['vocab_size']}


M = ledger()
BODY = M['fixed_body'] + M['routed_per_selected_expert_across_layers'] * K


def expert_union(tokens, route):
    if route == 'same-experts':
        return float(K)
    if route == 'max-union':
        return float(min(E, tokens * K))
    assert route == 'uniform-independent'
    return E * (1 - (1 - K / E) ** tokens)


def pairs(n, prefix=0, limit=None):
    if limit is None:
        return n * prefix + n * (n + 1) // 2
    # Sum min(limit, prefix+j), without a loop over long prompts.
    small = max(0, min(n, limit - prefix))
    return small * prefix + small * (small + 1) // 2 + (n - small) * limit


def resource_work(batch, n, prefix, decode, route):
    """A wave has batch requests, each executing n new tokens.

    Matrix ledger uses nonabsorbed logical projections. Attention distinguishes
    expanded prefill from absorbed decode; this is not a trace of the backend.
    HBM assumes perfect weight reuse within this one wave, no reuse across waves,
    balanced shards, and one-byte weights. Persistent on-chip weights, speculative
    decoding, different batching, and new representations change this model.
    """
    tokens = batch * n
    logits = batch  # one position per request, not every prefill position
    linear_ops = 2 * (tokens * BODY + logits * M['lm_head'])
    all_pairs = batch * pairs(n, prefix)
    selected_pairs = batch * pairs(n, prefix, C['index_topk'])
    index_ops = 2 * NI * C['index_n_heads'] * C['index_head_dim'] * all_pairs
    width = ((2 * C['kv_lora_rank'] + C['qk_rope_head_dim']) if decode
             else C['qk_nope_head_dim'] + C['qk_rope_head_dim'] + C['v_head_dim'])
    attention_bf16_ops = 2 * L * H * width * selected_pairs
    union = expert_union(tokens, route)
    weight_bytes = M['fixed_body'] + M['lm_head'] + M['routed_per_selected_expert_across_layers'] * union
    kv_write = tokens * KV * RANKS
    # Prefill internal traffic is omitted, allowing ideal on-chip reuse.
    # Decode reads one copy of selected latent state per rank and index keys.
    # Indexer work/read can be skipped by some paths; keep it labelled a scenario.
    kv_read = 0
    if decode:
        ctx = prefix + n
        comp = DATA['kv_components_per_token_per_rank']
        kv_read = batch * RANKS * (
            comp['mla_int8_packed'] * min(ctx, C['index_topk'])
            + (comp['indexer_bf16'] + comp['indexer_int8_with_scale']) * ctx)
    return {'linear_ops_relaxed_int8': linear_ops,
            'index_dot_ops_relaxed_int8': index_ops,
            'attention_ops_bf16': attention_bf16_ops,
            'expert_union': union, 'weight_bytes_relaxed': weight_bytes,
            'hbm_bytes_modelled': weight_bytes + kv_write + kv_read}


def cost(work, hw):
    # INT8/BF16 matrix paths share Cube capacity: add normalized service demands.
    cube = ((work['linear_ops_relaxed_int8'] + work['index_dot_ops_relaxed_int8']) /
            (RANKS * hw['int8_ops_s_per_die'])
            + work['attention_ops_bf16'] / (RANKS * hw['bf16_ops_s_per_die']))
    memory = work['hbm_bytes_modelled'] / BW
    return {'cube_s': cube, 'hbm_s': memory, 'roof_s': max(cube, memory)}


def scenario(i, o, b, route, hw):
    p = cost(resource_work(b, i, 0, False, route), hw)
    # Observed naive protocol: P's first token is discarded. D loads I-128
    # tokens and computes a 128-token suffix before returning its first token.
    tail = min(128, i)
    bootstrap = cost(resource_work(b, tail, i - tail, False, route), hw)
    steps = [cost(resource_work(b, 1, i + j - 1, True, route), hw)
             for j in range(1, o)]
    decode_s = sum(x['roof_s'] for x in steps)
    ttft = p['roof_s'] + bootstrap['roof_s']
    wave = ttft + decode_s
    # Synchronous-wave model only. It is not an upper bound over all possible
    # continuous-batching schedules. No queue/KV transport time is included.
    pd_period = max(p['roof_s'], bootstrap['roof_s'] + decode_s)
    kv_slots = b * (((i + o + 127) // 128) * 128)
    return {'input_tokens': i, 'output_tokens': o, 'decode_batch': b,
            'route': route, 'hardware_reference': hw['name'],
            'p_pass_roof_ms': p['roof_s'] * 1000,
            'pd_ttft_roof_ms_without_transfer': ttft * 1000,
            'decode_tpot_roof_ms_mean': decode_s / (o - 1) * 1000,
            'wave_output_tps_roof': b * o / wave,
            'wave_input_tps_roof': b * i / wave,
            'pipelined_wave_output_tps_roof': b * o / pd_period,
            'pipelined_wave_input_tps_roof': b * i / pd_period,
            'p_input_tps_roof': b * i / p['roof_s'],
            'decode_only_output_tps_roof': b * (o - 1) / decode_s,
            'kv_token_slots_needed_per_rank': kv_slots,
            'fits_snapshot_D_KV_slots_only': kv_slots <= 1289 * 128,
            'within_current_admission_settings': b <= 4 and i + o <= 8192}


def verify():
    assert NI == 21 and ND == 3 and NM == 75
    assert sum(DATA['kv_components_per_token_per_rank'].values()) == KV == 54528
    assert BODY + M['lm_head'] == 40297758720
    assert expert_union(1, 'uniform-independent') == K
    assert expert_union(1, 'max-union') == K
    assert pairs(1024, 100, 2048) == pairs(1024, 100)
    assert pairs(1, 4096, 2048) == 2048
    assert pairs(4096, 0, 2048) == sum(min(j, 2048) for j in range(1, 4097))
    for b in (1, 4, 16, 64, 128):
        assert K <= expert_union(b, 'uniform-independent') <= min(E, b * K)
    # Increasing matrix demand cannot improve the modeled resource time.
    w1 = resource_work(1, 1024, 0, False, 'uniform-independent')
    w2 = resource_work(1, 2048, 0, False, 'uniform-independent')
    assert all(w2[k] >= w1[k] for k in ('hbm_bytes_modelled', 'linear_ops_relaxed_int8'))
    print('PASS: units/ledger, packed KV, expert union, causal/sparse pairs, monotonic resource demand')


def main():
    verify()
    rows = [scenario(i, o, b, route, hw)
            for i, o in ((256, 64), (1024, 64), (4096, 128), (8192, 1024))
            for b in (1, 4, 16, 64, 128)
            for route in DATA['routing_scenarios']
            for hw in DATA['hardware_scenarios']]
    result = {'classification': 'conditional optimistic resource model; not measured, not certified global bounds',
              'inputs_sha256': hashlib.sha256((ROOT / 'inputs.json').read_bytes()).hexdigest(),
              'logical_matrix_ledger': M, 'body_ops_per_new_token': 2 * BODY,
              'output_token_matrix_ops': 2 * (BODY + M['lm_head']),
              'capacity_snapshot': {'P_KV_token_slots_per_rank': 1108 * 128,
                                    'D_KV_token_slots_per_rank': 1289 * 128,
                                    'current_context_limit': DATA['current_max_model_len'],
                                    'model_config_max_position': C['max_position_embeddings']},
              'rows': rows}
    (ROOT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    with (ROOT / 'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if (row['route'] == 'uniform-independent' and row['hardware_reference'] == '800T-paper-v1-reference'
                and row['decode_batch'] in (1, 4, 16, 64)):
            print(json.dumps(row))


if __name__ == '__main__':
    main()
