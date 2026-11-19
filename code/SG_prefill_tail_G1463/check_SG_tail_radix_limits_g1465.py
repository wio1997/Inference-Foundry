"""Check the fixed SGLang radix match limit when late-layer tail replay is enabled."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace


PREFILL_RECEIPT = '4cea5741b43eed04880917291656aa15a9944cde3f00902945b0a4127d530eb5'
DECODE_RECEIPT = 'bcf520726ded90e25950667170b36e115cb3ba50681b1d9e4584f75ef6f1562c'
MEMORY = 'mem_cache/deepseek_v4_memory_pool.py'
RADIX = 'mem_cache/unified_radix_cache.py'
KEY = 'mem_cache/common.py'
BUILDER = 'mem_cache/kv_cache_builder.py'
GEOMETRY = 'code/eager_diagnostics_G1113/native_attn_metadata_kernels.py'
GEOMETRY_SHA = '355e7e09320abd180d8886b1e1cc3fa4df4ed03e77175a3555f95825e9df04f3'
MARKER = 'G1465_SG_TAIL_CACHE '


def pinned_receipt(path, digest, role):
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == digest
    packet = json.loads(raw)
    assert packet['exitstatus'] == 0 and packet['outcome'] == 'finished'
    line, = (x.removeprefix(MARKER) for x in packet['output'].splitlines() if x.startswith(MARKER))
    payload = json.loads(line)
    assert payload['epoch'] == 1317 and payload['role'] == role
    assert payload['replay_resolution'] == ['False']
    return payload, packet['script_sha256']


def load_policies(backend, key_builder):
    # Use source functions from successful reads of the live immutable image.
    # Supply only the unified-KV predicate dependency and the RadixKey boundary.
    tree = ast.parse(dedent(backend) + '\n\n' + dedent(key_builder))
    radix, key = tree.body
    assert radix.name == 'swa_reprefill_tail_tokens' and key.name == '_req_radix_key'
    (original_import,) = (node for node in radix.body if isinstance(node, ast.ImportFrom))
    assert original_import.module.endswith('unified_kv_gate')
    assert [alias.name for alias in original_import.names] == ['is_unified_kv_triton']
    radix.body.remove(original_import)
    tree = ast.fix_missing_locations(tree)
    namespace = {'RadixKey': SimpleNamespace, 'TREE_WINDOW': 'window',
                 'is_unified_kv_triton': lambda: False}
    exec(compile(tree, '<fixed-A2-radix-policies>', 'exec', flags=annotations.compiler_flag), namespace)
    return namespace[radix.name], namespace[key.name]


def audit_cache_limit(repository, prefill, decode):
    sources = prefill['source_hits']
    assert all(sources[name]['sha256'] == decode['source_hits'][name]['sha256']
               for name in (MEMORY, RADIX, KEY, BUILDER))
    assert 'decoder_swa_bounded_replay=get_exec().features.enable_decoder_swa_bounded_replay' in (
        ''.join(line for _, line in sources[BUILDER]['selected']))
    memory_path = repository / 'code/native_radix_G491/fixed_source/sglang/srt' / MEMORY
    memory_raw = memory_path.read_bytes()
    assert sha256(memory_raw).hexdigest() == sources[MEMORY]['sha256']
    memory = memory_raw.decode()
    assert 'encoder_replay = get_exec().features.enable_encoder_swa_bounded_replay' in memory
    assert 'if encoder_replay and not is_draft_worker:' in memory

    radix_function = sources[RADIX]['function']
    key_function = sources[KEY]['function']
    tail_tokens, make_key = load_policies(radix_function['text'], key_function['text'])
    request = SimpleNamespace(extra_key='opaque', cache_salt=None)
    cases = (
        ('baseline_capped', False, False, True, 32768, 32767, 0, 32767),
        ('tail_capped', True, False, True, 32768, 32767, 128, 32640),
        ('tail_without_external_cap', True, False, True, 32768, None, 128, 32640),
        ('tail_with_smaller_external_cap', True, False, True, 32768, 32000, 128, 32000),
        ('short_prompt_entirely_recomputed', True, False, True, 63, 62, 128, 0),
        ('same_session_record', True, True, True, 32768, 32767, 0, 32767),
        ('no_SWA_window', True, False, False, 32768, 32767, 0, 32767),
    )
    outcomes = {}
    for name, enabled, own, window, prompt, cap, expected_tail, expected_limit in cases:
        swa = SimpleNamespace(sliding_window_size=128) if window else None
        cache = SimpleNamespace(decoder_swa_bounded_replay=enabled,
                                component_of=lambda family: swa if family == 'window' else None,
                                session=SimpleNamespace(will_lend_record=lambda given: own and given is request))
        cache.swa_reprefill_tail_tokens = lambda req: tail_tokens(cache, req)
        key = make_key(cache, request, range(prompt), cap)
        tail = cache.swa_reprefill_tail_tokens(request)
        assert (tail, key.limit) == (expected_tail, expected_limit), name
        outcomes[name] = {'prompt_tokens': prompt, 'external_max_prefix': cap,
                          'source_reprefill_tail_tokens': tail, 'radix_key_limit_before_page_matching': key.limit}
    return outcomes


def audit_geometry(repository):
    source = (repository / GEOMETRY).read_bytes()
    assert sha256(source).hexdigest() == GEOMETRY_SHA
    names = {'late_layer_tail_layout', '_whole_extend_tail_layout', 'build_causal_swa_page_indices'}
    nodes = {node.name: node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names}
    assert nodes.keys() == names
    text = {name: ast.unparse(node) for name, node in nodes.items()}
    assert 'max(0, s - tail_len)' in text['late_layer_tail_layout']
    assert 'max(0, s - t - tail_len) if whole else max(0, s - tail_len)' in text['_whole_extend_tail_layout']
    assert 'offsets < swa_replay_start.to(offsets.dtype).unsqueeze(1)' in text['build_causal_swa_page_indices']
    assert 'raw_indices.masked_fill_(invalid_offset_mask, -1)' in text['build_causal_swa_page_indices']

    geometry = {}
    window = 128
    for name, context, chunk, whole in (
        ('full_chunk_tail', 0, 8192, False),
        ('short_tail_after_full_chunk', 8192, 15, False),
        ('short_whole_extend_reader', 8192, 15, True),
    ):
        endpoint = context + chunk
        kept = chunk if whole else min(window, chunk)
        floor = max(0, endpoint - window - kept) if whole else max(0, endpoint - window)
        first_q = endpoint - kept
        first_visible = first_q - max(0, floor, first_q - window + 1) + 1
        geometry[name] = {'kept_query_rows': kept, 'floor': floor, 'first_query_position': first_q,
                          'causal_K_rows_at_first_query_if_resident': first_visible}
    assert [row['causal_K_rows_at_first_query_if_resident'] for row in geometry.values()] == [1, 114, 128]
    return geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    parser.add_argument('prefill_source_receipt', type=Path)
    parser.add_argument('decode_source_receipt', type=Path)
    args = parser.parse_args()
    p, p_observer = pinned_receipt(args.prefill_source_receipt, PREFILL_RECEIPT, 'prefill')
    d, d_observer = pinned_receipt(args.decode_source_receipt, DECODE_RECEIPT, 'decode')
    assert p['image'] == d['image'] and p_observer == d_observer
    startup = {}
    for data in (p, d):
        target = Counter((kind, allocator) for worker, kind, allocator in data['pool_types_in_startup_log']
                         if worker == 'target')
        assert set(target) == {('paged', 'True')} and target[('paged', 'True')] >= 8
        startup[data['role']] = {'target_SWA': 'paged', 'matching_log_lines': target[('paged', 'True')],
                                 'server_prefix_sha256': data['server_start_prefix_sha256']}
    matches = audit_cache_limit(args.repository, p, d)
    geometry = audit_geometry(args.repository)
    result = {
        'epoch': 1317, 'fixed_image': p['image'], 'baseline_startup': startup,
        'CPU_fixed_source_radix_controls': matches,
        'source_implied_swa_geometry_when_tail_is_enabled': geometry,
        'actual_tail_candidate_cache_or_NPU_verified': False,
        'source_SHA256': {name: p['source_hits'][name]['sha256'] for name in (MEMORY, RADIX, KEY, BUILDER)},
        'geometry_source_SHA256': GEOMETRY_SHA,
        'source_receipt_SHA256': {'prefill': PREFILL_RECEIPT, 'decode': DECODE_RECEIPT,
                                  'observer_source': p_observer},
        'scope': ('Actual baseline storage logs and fixed source CPU admission policy with a non-unified KV stub. '
                  'Radix limits are before page matching. SWA calculations describe logical resident rows. '
                  'Device KV bytes, page mapping, cache hits and E2E are pending.'),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
