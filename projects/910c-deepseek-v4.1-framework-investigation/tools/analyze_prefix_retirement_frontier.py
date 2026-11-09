#!/usr/bin/env python3
"""Bound current guarded recipes and replay pure ownership counterexamples.

Original methods run with inert allocator/key fixtures. No Torch/NPU import,
model output comparison, native acceptance or measured request speedup.
"""
import ast
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

OUT = Path('audit/execution-search')


def load(name):
    return json.loads((OUT / name).read_text())


def extract(path, names, cls=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body if cls is None else next(
        n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    chosen = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in chosen} == set(names)
    return ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',
                                          names=[ast.alias(name='annotations')], level=0),
                            *chosen], type_ignores=[]))


def main():
    identity = load('prefix-retirement-execution-identity.json')
    extra = load('prefix-retirement-extra-dependencies.json')
    rows = identity['sources'] + extra['sources']
    by = {r['runtime_path'].removeprefix('/vllm-workspace/'): r for r in rows}
    for r in rows:
        assert sha256(Path(r['local_path']).read_bytes()).hexdigest() == r['runtime_sha256']
    profile = load('prefix-retirement-profile.json')
    old = {r['rank']: r for r in load('inactive-dp-framework-edges-receipt.json')['files']}
    assert set(r['rank'] for r in profile['files']) == set(range(16))
    assert all(r['sha256'] == old[r['rank']]['sha256'] and r['bytes'] == old[r['rank']]['bytes']
               for r in profile['files'])
    config = load('prefix-retirement-config-record.json')
    assert identity['saved_launch']['sha256'] == config['saved_serve']['sha256']
    options = identity['saved_launch']['selected_options']
    assert options['--async-scheduling'] is True
    assert json.loads(options['--speculative-config'])['method'] == 'dspark'
    assert options['--block-size'] == '128'
    absent = config['saved_serve']['absent_whitelisted_options']
    assert all(k in absent for k in ['--kv-transfer-config', '--ec-transfer-config',
                                     '--kv-offloading-size', '--cpu-offload-gb',
                                     '--offload-group-size'])
    additional = json.loads(options['--additional-config'])
    assert 'sparse_kv_offload_config' not in additional
    assert json.loads(options['--engram-config'])['cpu_offload'] is True
    pool_path = Path(by['vllm/vllm/v1/core/block_pool.py']['local_path'])
    env = dict(make_block_hash_with_group_id=lambda h, g: (h, g))
    exec(compile(extract(pool_path, ['free_blocks', 'get_cached_block',
                                    'is_block_writable'], 'BlockPool'), str(pool_path), 'exec'), env)

    class Queue:
        def __init__(self):
            self.rows = []

        def prepend_n(self, rows):
            self.rows = list(rows) + self.rows

        def append_n(self, rows):
            self.rows.extend(rows)

    block = SimpleNamespace(block_id=7, ref_cnt=1, is_null=False,
                            block_hash=('prefix-H', 0), pool=None)
    table = {('prefix-H', 0): block}
    pool = SimpleNamespace(enable_caching=True, free_block_queue=Queue(),
                           cached_block_hash_to_block=SimpleNamespace(get_one_block=table.get))
    env['free_blocks'](pool, [block])
    match = env['get_cached_block'](pool, 'prefix-H', [0])
    assert block.ref_cnt == 0 and match == [block]
    assert pool.free_block_queue.rows == [block]
    assert not env['is_block_writable'](pool, block)
    free_witness = dict(ref_cnt_after_free=block.ref_cnt, still_matches_same_prefix=True,
                        in_free_queue=True, is_block_writable=False,
                        implication='Refcnt0/free-list is an eviction candidate, not proof of dead cache bytes; do not erase/alias before reuse invalidation and device lease retirement.')

    hash_path = Path(by['vllm/vllm/v1/core/kv_cache_utils.py']['local_path'])
    henv = dict(BlockHash=lambda x: x, NONE_HASH=b'fixture-seed')
    exec(compile(extract(hash_path, ['hash_block_tokens']), str(hash_path), 'exec'), henv)
    # The injected digest serializer is an inert fixture; dependency, not exact
    # vLLM cache key bytes or collision behavior, is under examination.
    h = henv['hash_block_tokens']
    digest = lambda obj: sha256(repr(obj).encode()).digest()
    token_ids = [1, 2, 3]
    a = h(digest, b'parent-A', token_ids, None)
    b = h(digest, b'parent-B', token_ids, None)
    c = h(digest, b'parent-A', token_ids, ('salt-or-adapter',))
    assert len({a, b, c}) == 3
    cache_labels = [dict(rank=r['rank'], cache_specific_events=len(r['cache_specific_events']),
                         prepare_input_labels=sum(e['name'] == 'prepare input' for e in r['parents']))
                    for r in profile['files']]
    # Source conditions, not absence of labels, make these GPU recipes empty.
    zero = dict(recipe='Fuse/remove current per-request AscendKVBlockZeroer GPU work',
                state='已停止', original_gpu_work=0, favourable_local_gpu_saved_ms=0,
                conditional_request_saved_ms=0,
                source='worker.py:1208-1221: MRV1 plus DSpark fails V2-or-Mamba/Eagle3 init guard; _zero_block_ids calls absent object through hasattr; C8 MXFP scale-only list excludes DSV4.1 MLA/SWA ABI.',
                limits='Does not remove ordinary aten zero/fill, C2 ring writes or startup tensor zeros. CPU ID drain/build/bookkeeping still exists and overlaps host/BatchFrame scope; no numerical native acceptance.')
    cow = dict(recipe='Fuse current partial-local-hit CoW copy stack and GPU copy',
               state='已停止', original_gpu_work=0, favourable_local_gpu_saved_ms=0,
               conditional_request_saved_ms=0,
               source='No Mamba spec in fixed DSV4.1 cache topology; hit alignment stays scheduler LCM128. Cacheable MLA/SWA groups block128, noncacheable circular32. No within-block prefix tail for _has_partial_local_hit; no remote producer/consumer partial Mamba tail.',
               limits='New partial-hit/config/layout mechanisms must be re-evaluated; no general statement that CoW is unnecessary.')
    out = dict(status='PARTIAL_PREFIX_CACHE_PROVIDER_AND_RETIREMENT_REVIEW',
               whole_chain_complete=False, precision='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',
               primary_sources=len(identity['sources']), extra_sources=len(extra['sources']),
               current_prior_sha_matches=sum(r.get('matches_local', False) for r in rows),
               config_contract=dict(prefix_caching=True, async_scheduling=True,
                                    ordinary_KV_transfer=None, EC_transfer=None,
                                    generic_weight_offload='Noop', sparse_KV_offload=False,
                                    Engram_CPU_table_offload=True,
                                    record='Completed profile startup and unchanged saved launch/default source; not live object dump'),
               source_findings=[
                   'Request hashes extend only on new full blocks and chain parent/token/extra identity; do not credit full-prefix rehash every decode.',
                   'Hybrid lookup groups equal specs/managers/pools, full-attention downward prefix is trimmed on later fixed-point iterations. Existing sharing must not be double-counted.',
                   'Prefix lookup caps num_tokens-1 for logits; DSpark use_eagle() True and default trailing drop retained. No-Mamba partial hits disabled. Circular state is request-private/noncacheable.',
                   'All-group cache-hit references are touched before external allocations; local normal allocation then new blocks cannot evict unretained hits.',
                   'Only committed/finalized tokens publish hashes; async placeholders, in-flight tokens and rejection rollback remain separate.',
                   'Finished/abort/forced-preempt/reset and reused request ID require worker tombstone cleanup before new request installation; empty execute_model still applies _update_states before returning.',
                   'Current defer_block_free is False without KV consumer connector; CPU block return relies on original worker device-ordering. New async transfer/early-input/megakernel needs explicit leases, not an invented currently-enabled fence.',
                   'Cache free keeps hashes and contents; actual get_new_blocks evicts hash/alerts reuse watchers. Reset fails while non-null blocks remain referenced.',
                   'Async speculative config disables cascade attention; generic two-softmax cascade is not a supported bit-exact flag flip.'
               ], original_methods_counterexamples=dict(free_does_not_kill_cache=free_witness,
                   hash_dependency=dict(same_token_block_different_parent_differs=True,
                                        same_parent_tokens_different_extra_differs=True,
                                        limits='Original function body with injected fixture digest/serializer; not actual SHA key equivalence or model numerical test.')),
               current_scoped_recipes=[zero, cow, dict(
                   recipe='Remove existing remote KV/EC or sparse offload transfer and generic weight-prefetch cost',
                   state='已停止', favourable_local_saved_ms=0,
                   limits='Absent on unchanged frozen serving; introducing a new owner/layout/offload design is a different candidate. Engram CPU lookup is live and excluded from this zero scope.'), dict(
                   recipe='Enable cascade/disable trailing prefix guard or substitute cached hidden across original M geometry',
                   state='已停止', limits='Flag-only/unqualified generic recipe: async-SD disables cascade, volatile draft tail and original primitive geometry/rounding cannot be assumed away. Larger original-geometry shared-read path pending.')],
               valuable_joint_frontiers=[dict(name='Typed physical-cache ownership + immutable BatchFrame + retirement protocol',
                   state='待确认', relation='overlaps existing metadata-host, whole-inactiveDP and physical cache scope; no additive price',
                   gap='Whole engine scheduling/lookup/ID bookkeeping CPU parent not recorded by worker traces; original all-reader/range generation/device-last-reader and new pin/event/storage costs required. Do not copy old worker-only boundary into this wider parent.'), dict(
                   name='Cross-request shared prefix/selected-KV loading under original row arithmetic', state='待确认',
                   relation='overlaps existing SMLA/QLI unique-row ownership and physical cache producer; not a newly measured gain',
                   gap='Actual per-rank selected physical row overlap, source owner generation/refcount, simultaneous live Q needs, existing L2 sharing and original softmax/reduction geometry; fixed C8 existing hit already saves old prefix compute. Six ordinary loads need matched populations/times.')],
               new_implementable_admission=[], profile_cache_label_navigation=cache_labels,
               profile_qualification='All16 old trace hashes/sizes match; zero cache-specific and prepare-input labels in this capture do NOT independently prove no call. Source hard guards establish scoped absent GPU work. Generic zero/fill totals are not cache cost/net saving.',
               next_action='Continue unclosed startup/warmup/MM/provider dispatch/error callees, then source-coverage audit from actual call graph rather than historical opportunity inventory. Preserve valuable native/physical gates pending.',
               inputs=[dict(path=str(OUT / f), sha256=sha256((OUT / f).read_bytes()).hexdigest())
                       for f in ['prefix-retirement-execution-identity.json',
                                 'prefix-retirement-extra-dependencies.json',
                                 'prefix-retirement-config-record.json', 'prefix-retirement-profile.json']],
               analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
               new_NPU_work=0, new_profile_collection=0, service_requests=0, remote_writes=[])
    (OUT / 'prefix-retirement-frontier.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(current_sources=len(rows), prior_hash_matches=out['current_prior_sha_matches'],
                          counterexample=free_witness, source_guarded_GPU_prices_ms=[0, 0],
                          new_native_admission=[], whole_chain_complete=False)))


if __name__ == '__main__':
    main()
