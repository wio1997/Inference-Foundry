#!/usr/bin/env python3
"""Source-qualified staging ownership frontier; no runtime imports or experiments."""
import ast
import hashlib
import json
from pathlib import Path

OUT = Path('audit/execution-search')

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    identity_path = OUT / 'staging-execution-identity.json'
    identity = json.loads(identity_path.read_bytes())
    sampler_identity = json.loads((OUT/'sampler-execution-identity.json').read_bytes())
    sampler_sources = {s['runtime_path']: s for s in sampler_identity['source_pins']}
    extra_reviews = []
    for name, ranges in [('vllm-ascend/vllm_ascend/ops/triton/reject_sample.py', [[36, 168]]),
                         ('vllm-ascend/vllm_ascend/sample/rejection_sampler.py', [[174, 242], [424, 516]])]:
        s = sampler_sources['/vllm-workspace/'+name]
        assert digest(s['local_path']) == s['runtime_sha256']
        extra_reviews.append(dict(runtime_path=s['runtime_path'], sha256=s['runtime_sha256'], ranges=ranges))
    sources = {s['runtime_path'].removeprefix('/vllm-workspace/'): s for s in identity['sources']}
    for s in sources.values():
        assert digest(s['local_path']) == s['runtime_sha256']

    def source(name):
        return Path(sources[name]['local_path']).read_text()

    runner = 'vllm-ascend/vllm_ascend/worker/model_runner_v1.py'
    parent = 'vllm/vllm/v1/worker/gpu_model_runner.py'
    proposer = 'vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py'
    tree = ast.parse(source(runner))
    scopes = sorted([dict(start=n.lineno, end=n.end_lineno)
              for n in ast.walk(tree) if isinstance(n, ast.With)
              and any('synchronize_input_prep' in ast.unparse(i.context_expr) for i in n.items)], key=lambda n: n['start'])
    assert scopes == [dict(start=2288, end=2514), dict(start=4045, end=4180)]
    assert 'self.prepare_inputs_event.synchronize()' in source(parent)
    assert 'self.prepare_inputs_event.record()' in source(parent)
    assert 'return True' in source('vllm-ascend/vllm_ascend/platform.py').split('def is_pin_memory_available(cls):')[1].split('\n\n')[0]
    assert source(proposer).splitlines()[2216].strip() == 'self.backup_next_token_ids.copy_to_gpu(num_reqs)'

    reviewed = {
        runner: [[379, 423], [611, 645], [825, 885], [984, 985], [1281, 1356],
                 [1437, 1543], [1575, 1610], [1737, 1762], [2286, 2340],
                 [2418, 2514], [2516, 2548], [2764, 2970], [5993, 6010]],
        parent: [[775, 800], [1033, 1046], [3840, 3853]],
        proposer: [[780, 840], [1090, 1140], [1250, 1400], [1490, 1610],
                   [2192, 2252], [2295, 2365]],
        'vllm-ascend/vllm_ascend/worker/npu_input_batch.py': [[38, 258]],
        'vllm-ascend/vllm_ascend/worker/block_table.py': [[22, 194], [196, 250], [298, 337], [340, 447], [469, 529]],
        'vllm-ascend/vllm_ascend/worker/device_metadata.py': [[46, 155], [174, 252]],
        'vllm-ascend/vllm_ascend/worker/utils.py': [[239, 271]],
        'vllm-ascend/vllm_ascend/attention/dsa_v41.py': [[879, 927], [929, 1010], [1160, 1209]],
        'vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py': [[75, 146]],
        'vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py': [[1146, 1218]],
        'vllm/vllm/v1/spec_decode/llm_base_proposer.py': [[142, 153], [248, 265], [515, 610]],
        'vllm/vllm/v1/utils.py': [[110, 150]],
        'vllm/vllm/utils/torch_utils.py': [[21, 21], [75, 75]],
        'vllm/vllm/utils/platform_utils.py': [[43, 47]],
        'vllm-ascend/vllm_ascend/platform.py': [[146, 148]],
    }
    inventory = [
        dict(owner='runner + NPUInputBatch', storage='CpuGpuBuffer.cpu/.np; input_ids; num_computed_tokens_cpu_tensor',
             mutate='execute_model input scope, including persistent request updates',
             last_read='target preparation H2D enqueued before scope exit',
             existing_protection='previous prepare_inputs_event host synchronize before any protected CPU mutation; current event recorded after input preparation',
             gap='does not fence later proposer H2D; device readers and foreign branches need separate qualification'),
        dict(owner='Ascend BlockTable', storage='persistent INT32 block_table.cpu; NumPy shares storage',
             mutate='append/add/clear/move/swap rows', last_read='commit_block_table(num_reqs) copies whole selected rows, including allocated width',
             existing_protection='runner prepare scope covers _update_states and _prepare_inputs/commit',
             gap='draft CPU slot_mapping fallback and CP/graph branches must not inherit eager DSpark guard'),
        dict(owner='AscendKVBlockZeroer', storage='_ids_pinned INT64 grow-only capacity; overwritten prefix',
             mutate='zero_block_ids called in _update_states', last_read='idx.copy_(pinned[:n], non_blocking=True) before zero kernel',
             existing_protection='inside the same runner prepare scope',
             gap='a2 tests no-auto-snapshot backend rule, not a race in this protected unchanged service'),
        dict(owner='DSpark proposer', storage='backup_next_token_ids.cpu/.np; INT32[max_num_seqs=4], capacity16B/rank',
             mutate='prepare_next_token_ids_padded at2214', last_read='copy_to_gpu(num_reqs) at2217, after _sample/bookkeeping and outside recorded target preparation',
             existing_protection='not covered by the same target preparation event',
             gap='needs its own read-complete lease or proved dead fallback; allocation size is not total lease cost'),
        dict(owner='V41 CP metadata builder', storage='fresh qsl/suffix results then pin_memory, not persistent parent CPU storage',
             mutate='CPU clamp/subtract produce new results; no later in-place stores in reviewed path',
             last_read='query_start_loc.copy_(qsl.pin_memory()) and suffix.pin_memory().to(device)',
             existing_protection='per-call immutable source rather than shared mutable CPU storage',
             gap='no claim that allocator lifetime or all delegated consumers have native admission'),
        dict(owner='TargetDeviceMetadata / DeviceMetadataExecutor', storage='persistent device metadata buffers and event frontiers; per-build shared dictionaries',
             mutate='target builders / metadata stream', last_read='stage waits joined by TargetDeviceMetadata.finish; reuse fence is a device event',
             existing_protection='device dependency protocol, not a CPU source lease',
             gap='graph, dummy/real, empty DP and draft release need integration qualification; dictionaries are not proved cross-iteration content caches'),
    ]
    d = dict(status='PARTIAL_SOURCE_OWNERSHIP_AUDIT', completeness_established=False,
             precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',
             source_identity=dict(path=str(identity_path), sha256=digest(identity_path), sources=len(sources)),
             reviewed_ranges=[dict(runtime_path='/vllm-workspace/'+name, ranges=ranges,
                                   sha256=sources[name]['runtime_sha256']) for name, ranges in reviewed.items()],
             fallback_source_reviews=extra_reviews,
             existing_preparation_scopes=scopes, inventory=inventory,
             correction='All-input double buffering is not established as necessary. Existing host preparation event already protects target mutable CPU staging; known later persistent source is proposer backup. Native no-snapshot finding remains valid within that narrower ownership model.',
             alternatives=[
                 dict(name='early D2H plus extended preparation fence', status='待确认',
                      design='re-record a reuse event after draft H2D/required joins; a conservative draft-end event keeps CPU source ownership without a ring while allowing output thread to deliver early',
                      cost='must charge lost next-input CPU preparation overlap and any driver contention; cannot award throughput or preserve old response-end time for free',
                      gap='prove exact final H2D frontier, event ordering across all used streams, and matched-load host wait cost'),
                 dict(name='early D2H plus no-discard greedy dead-backup path', status='待确认',
                      design='guard no discarded requests and actual greedy verifier; retain original selected IDs/counts and torch.where, omit only dead CPU fallback fill/H2D when every row has >=1 valid token',
                      cost='16B capacity/rank does not explain value; value is removing the remaining persistent-source dependency, not bandwidth',
                      source_proof='current ordinary greedy returns a vocabulary argmax. Current general greedy verifier writes the first target argmax when K>0 and writes bonus at column0 when K=0; special K1 also writes column0. With no discard the clone/index_fill cannot erase it, so valid_count>=1 and torch.where never selects backup. This is a conditional source proof, not a native replacement result.',
                      gap='qualify legal full-batch eligibility including prefill/dummy/empty and full caller set, audit delegated late readers, then native byte equality and integration'),
                 dict(name='immutable snapshot / ring for live late sources', status='待确认',
                      design='lease only actual live physical source ranges until last DMA completion; retain original fallback on branches not qualified',
                      cost='allocation/copy/event/wait/backpressure and possible CPU scheduling loss must be paid; no blanket all-input multiplication',
                      gap='concrete source set and observed reuse pressure, not another automatic-snapshot probe')],
             pricing=dict(inherited_component_budget='7–15ms first eager reply; short slot7 15.238845ms /3.395119757%, P90 8.01552ms /0.537285518%',
                          net_speedup_established=False, eligible_request_set_established=False,
                          admission='price guard-eligible subset and added protocol/host scheduling costs before claiming net gain; ordinary six loads have no matched early-output measurement'),
             archived_eligibility=dict(path=str(OUT/'staging-archived-eligibility-schema.json'),
                 qualification='Two original reply-worker logs SHA checked, rank0 has2641 and rank8 has2677 records. Schemas do not record num_discarded_requests or all_greedy. Scheduling membership alone is insufficient to prove all rows valid. No new inference/profile run authorized by this absence; keep specific eligibility gap.'),
             native_work_this_review=0, service_requests=0, new_profile_collection=0, remote_writes=[],
             next_action='Price shared metadata/host submission as the next larger parent, preserving the concrete early-output eligibility and full-caller/stream gaps. The no-discard greedy dead-backup source implication is derived; archived reply logs lack the deciding flag. Do not repeat snapshot tests or assume a whole-input ring. Revisit final-layer attention/FFN producer after this parent.')
    p=OUT/'staging-execution-frontier.json'
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=d['status'], sources=len(sources), inventory_groups=len(inventory), native_work=0, completeness_established=False)))

if __name__ == '__main__':
    main()
