#!/usr/bin/env python3
"""Reconcile read-only startup sources, completed logs and weight HEADERs.

No framework imports, model/device work, service calls or remote writes.
Source review ranges are annotations of a human/agent review, not inferred
from a file hash or a function-name search.
"""
import ast
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from types import SimpleNamespace

OUT = Path('audit/execution-search')


def read(name):
    return json.loads((OUT / name).read_text())


def pin(path):
    p = Path(path)
    return {'path': str(p), 'sha256': sha256(p.read_bytes()).hexdigest()}


def original_function(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    evidence_names = ['startup-warmup-execution-identity.json',
                      'startup-helper-execution-identity.json',
                      'startup-helper-dependencies.json',
                      'startup-serving-consumers.json', 'startup-scope-evidence.json',
                      'startup-vision-header-bank.json']
    artifacts = [pin(OUT / name) for name in evidence_names]
    sources = {}
    for name in evidence_names[:-1]:
        for row in read(name).get('sources', []):
            expected = row.get('runtime_sha256', row.get('sha256'))
            actual = pin(row['local_path'])
            assert actual['sha256'] == expected, (name, row['runtime_path'])
            sources[row['runtime_path']] = {**actual, 'runtime_path': row['runtime_path']}
    scope = read('startup-scope-evidence.json')
    header = read('startup-vision-header-bank.json')
    log = scope['completed_startup_log']
    ranks = defaultdict(dict)
    for row in log['selected']:
        text = row['text']
        m = re.search(r'Worker_DP(\d+)_TP(\d+)_EP(\d+) pid=(\d+)', text)
        if not m:
            continue
        dp, tp, ep, pid = map(int, m.groups())
        assert ep == dp * 8 + tp
        key = f'DP{dp}_TP{tp}_EP{ep}'
        ranks[key]['pid'] = pid
        helper = re.search(r'(rejection_sampler|penalties|rms|indexer) Triton warmup complete in ([\d.]+)s', text)
        total = re.search(r'Triton kernel warmup finished in ([\d.]+)s', text)
        if helper:
            assert helper[1] not in ranks[key]
            ranks[key][helper[1]] = float(helper[2])
        if total:
            assert 'total' not in ranks[key]
            ranks[key]['total'] = float(total[1])
    assert len(ranks) == 16
    fields = ['rejection_sampler', 'penalties', 'rms', 'indexer', 'total']
    assert all(set(fields) <= set(row) for row in ranks.values())
    warmup = {
        'completed_diagnostic_log': {k: log[k] for k in ['path', 'sha256', 'bytes', 'mtime_ns']},
        'rank_rows': dict(sorted(ranks.items())),
        'seconds_by_helper': {k: {'min_rank': min(r[k] for r in ranks.values()),
                                  'max_rank': max(r[k] for r in ranks.values())} for k in fields},
        'scope': 'Rounded perf_counter wrapper elapsed seconds in an already completed diagnostic. Parallel rank totals are NOT summed; max is an envelope, not causal removable cold latency. No warmed-request timing credit.',
    }
    cfg = scope['model_config']['raw_config']
    text, vision = cfg['text_config'], cfg['vision_config']
    d, inner, layers = vision['hidden_size'], vision['intermediate_size'], vision['num_hidden_layers']
    h = text['hidden_size']
    patch = (3 * vision['patch_size'] ** 2 + 1) * d
    block = 2*d + (d*3*d+3*d) + (d*d+d) + d*2*inner + inner*d
    align_in = d * vision['downsample_ratio'] ** 2
    aligner = (align_in*h+h) + (h*h+h)
    expected_numel = patch + layers*block + d + aligner + 3*h
    selected = header['selected']
    assert len(selected) == 266 and len({r['parameter'] for r in selected}) == 266
    assert sum(r['numel'] for r in selected) == header['total_numel'] == expected_numel == 485268480
    assert sum(r['payload_bytes'] for r in selected) == header['total_checkpoint_payload_bytes'] == 970536960
    assert {r['dtype'] for r in selected} == {'BF16'} and header['tensor_payload_bytes_read'] == 0
    source_bf16_and_fp32_delimiters = 2 * (expected_numel - 3*h) + 4 * 3*h
    bank = {
        'scope': 'Exact checkpoint tensor shape/dtype payload under original _vision_parameter_name AST. No tensor bytes read. Source constructor predicts BF16 vision/aligner and explicit FP32 delimiters; no live storage/allocator observation.',
        'header_files': len(header['headers']), 'header_bytes': sum(r['header_bytes'] for r in header['headers']),
        'selected_tensors': len(selected), 'parameters_by_source_formula':
            {'patch': patch, 'block': block, 'blocks': layers, 'final_norm': d, 'aligner': aligner, 'delimiters': 3*h},
        'total_numel': expected_numel, 'checkpoint_dtype': 'BF16',
        'checkpoint_payload_bytes': header['total_checkpoint_payload_bytes'],
        'checkpoint_payload_MiB': header['total_checkpoint_payload_bytes']/2**20,
        'source_dtype_payload_bytes': source_bf16_and_fp32_delimiters,
        'source_dtype_payload_MiB': source_bf16_and_fp32_delimiters/2**20,
        'all_FP32_payload_ceiling_MiB': expected_numel*4/2**20,
        'historical_full_owner_OA_bank_MiB_per_rank': 2240,
        'source_dtype_payload_gap_to_that_bank_MiB': 2240-source_bf16_and_fp32_delimiters/2**20,
        'qualification': 'Sole declared-parameter payload bank is insufficient for that 2240 MiB scheme even if every parameter were FP32. This is NOT a physical VRAM upper bound: allocator padding, non-parameter/native buffers, profile peak and graph memory remain separate. Joint donors/old headroom are not excluded.',
    }
    budget_source = next(v['path'] for k,v in sources.items() if k.endswith('/encoder_cache_manager.py'))
    fn = original_function(budget_source, 'compute_mm_encoder_budget',
                           {'logger': SimpleNamespace(warning=lambda *a: None)})
    fixed = SimpleNamespace(disable_chunked_mm_input=False,
                            max_num_batched_tokens=8192, max_num_encoder_input_tokens=8192,
                            encoder_cache_size=8192)
    assert fn(fixed, {'image': 1024}) == (8192,8192)
    assert fn(fixed, {}) == (0,0)
    undersized = SimpleNamespace(disable_chunked_mm_input=True,
                                 max_num_batched_tokens=512, max_num_encoder_input_tokens=512,
                                 encoder_cache_size=512)
    try:
        fn(undersized, {'image': 1024})
    except ValueError:
        pass
    else:
        raise AssertionError('Original chunk-disabled exception absent')
    observed = [r for r in log['selected'] if 'profiled with 8 image items' in r['text']]
    assert len(observed) == 2 and all('8192 tokens' in r['text'] for r in observed)
    frontier = {
        'status': 'BOUNDED_STARTUP_SOURCE_LOG_AND_CAPACITY_REVIEW',
        'whole_chain_complete': False, 'bitwise_required_for_each_replacement': True,
        'evidence': artifacts, 'source_pins': list(sources.values()),
        'warmup': warmup, 'vision_parameter_bank': bank,
        'MM_budget': {
            'actual_completed_observation': observed,
            'original_compute_budget_fixture': {'fixed': [8192,8192], 'empty_active_modalities': [0,0], 'oversized_item_without_chunking': 'ValueError'},
            'fixed_shape_inputs': '32 vision layers, maximage1024; nested config flattened by DeepseekV41Config. Saved launch has no explicit language-only/limit-mm/skip-mm/kv-memory switch. Defaults are source-bound; completed log proves actual encoder profile for this diagnostic.',
            'consumers': ['GPUModelRunner.profile_run -> dummy multimodal inputs -> model.embed_multimodal -> encoder_cache tmp entries',
                          'maximum-batch dummy model + sampler -> synchronize -> encoder_cache.clear -> gc',
                          'NPUWorker.determine_available_memory: peak/non-KV accounting -> available KV capacity; explicit/startup-plan KV bytes still call profile_run',
                          'Serving text-only request: embed_multimodal has no image input, language input embeddings retained'],
            'pending': 'Any text-only tower removal must preserve config/MM-prefix behavior, original KV policy/capacity, weight loading and primitive bits; allocator/native peak/headroom impact and new net serving exposure are unmeasured.',
        },
        'helper_delivery': {
            'rejection': 'HAS_TRITON/spec/K guards; PP1 no_draft_probs=True; maxreq4 fits one BLOCK_SIZE bucket if V>=4, representative4, NOT 1/2/4. Prepare once, recovered per prob constexpr, expand two dtypes, greedy None/bool, random/block/entropy config variants; consumes RNG in dummy tensors. Does not by itself prove every K1/serving JIT key warmed.',
            'penalties': '4 x 16160 TP-local logits; prompt257 and neutral penalty values. Explicit warmup compiles path despite serving no_penalties early return in AscendSampler/RejectionSampler. Source neutral guard is not a generic IEEE algebraic identity.',
            'rms': 'Exact AscendDSABackend identity guard excludes V4.1 DeepseekV41CacheBackend. early assume_used branch exists but default early_kernel=False and no saved override. Rounded .000s in all16 ranks supports but does not replace guard proof. Adaptive UB launch in live triton_q_rms prevents treating fixed helper representatives as universal JIT completeness.',
            'indexer': 'Quant query(1,32,128), ratios1/2, prepare row-block reps1/2/4/8. Warmup lengths_output=None, WRITE_LENGTHS=False. Live source passes topk_lengths; topk512 fuses length only for rows<=1024, otherwise separate output>=0 + sum. Serving group membership and chronological index order remain live consumers.',
            'early': 'Early kernel and NZ defaultFalse; start/load/join traced. They are not already-overlapped savings in frozen defaults. Daemon exception fallback and600s timeout do not imply successful completion. NZ comment15-19s is not this workload measurement.',
        },
        'workspace': {
            'delivery': 'worker.init_device -> global manager(device,num_ubatches,default1lane); slot=(ubatch,lane), 256B aligned simultaneous views -> grow by replacing only requested slot before capture lock -> inherited GPU capture locks -> shutdown cleanup/reset',
            'hot_growth_credit_ms': 0,
            'proof_scope': 'After locked graph path, larger requests raise instead of hot reallocating. Current Ascend tree literal references contain init sites, no get_simultaneous consumer; generic CUDA/ROCm/etc consumers are not evidence of this model using the arena. No assertion of zero live native/ACL/DFC workspace.',
            'new_overlap': 'ContextVar lane reset is logical ownership, not device completion. New cross-stream/iteration sharing needs disjoint lanes or leases through actual final readers; EndCrossRankSync before Unpermute does not retire entire DFC arena.',
            'literal_reference_files': len(scope['workspace_references']),
        },
        'decisions': [
            {'id': 'startup-free-warm-request', 'category': '已停止', 'introduced': 'Delete/fuse cold helper and encoder preparation for warmed C8/ordinary request speed.', 'reached': 'Worker initialization, actual four helpers, early guards, profile_run/capture and cleanup, completed16-rank startup log.', 'result': 'Direct warmed request removable time0; same independent budget and child recipes stop. No cold-start speedup claim.'},
            {'id': 'locked-workspace-hot-realloc', 'category': '已停止', 'introduced': 'Remove repeated generic arena growth from captured hot path.', 'reached': 'init/slot allocator/grow/lock/inherited capture/reset and literal caller inventory.', 'result': 'Locked growth is illegal; no hot growth credit. Native workspace and new lifetime/overlap designs remain distinct.'},
            {'id': 'vision-memory-donor', 'category': '待确认', 'introduced': 'Use text-only unused vision capacity for a larger useful delivery bank.', 'reached': '32-layer constructor+all266 matched parameter headers; 485268480 elements, checkpoint925.576MiB, source dtype925.605MiB. MM budget really profiles8images before requests.', 'result': 'Sole parameter bank does not supply2240MiB/rank. Standalone warm compute deletion stops; joint capacity designs remain conditional on allocator/headroom/KVpolicy/bitwise behavior and matched request value.'},
            {'id': 'serving-indexer-delivery', 'category': '待确认', 'introduced': 'Whole query quantization -> QLI membership/candidates -> chronological indices/lengths -> shared SMLA consumers.', 'reached': 'Actual source caller/guards, FP16-scale quantizer, reversible FP sort, fused threshold, persistent output buffers and group sharing read. Cold helper credit excluded.', 'result': 'Next derive actual rows/ownership and matched whole parent budget, then deepen any worth-retaining native producer/consumer fusion. No new primitive admission or serving speedup.'},
        ],
        'next_action': 'Actual serving indexer quantization/QLI/canonical indices+lengths to shared SMLA: derive live per-rank rows and reuse scope, use existing matched profile to bound whole parent before small recipes; keep original arithmetic/ordering and every replacement bitwise.',
        'environment': {'new_NPU_work': 0, 'model_loads': 0, 'tensor_payload_bytes_read': 0,
                        'new_profile_collection': 0, 'service_requests': 0, 'remote_writes': [], 'service_changes': []},
    }
    (OUT/'startup-delivery-frontier.json').write_text(json.dumps(frontier,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'source_pins_verified':len(sources),'warmup_total_max_rank_s':warmup['seconds_by_helper']['total']['max_rank'],
                      'checkpoint_parameter_MiB':bank['checkpoint_payload_MiB'],
                      'source_dtype_parameter_MiB':bank['source_dtype_payload_MiB'],
                      'whole_chain_complete':False,'new_NPU_work':0},ensure_ascii=False))


if __name__ == '__main__':
    main()
