"""Source locations only; sampled Python ancestry is never exclusive CPU/wall."""
from pathlib import Path
import collections
import hashlib
import json
import re
import sys


def reduce(root):
    observation = json.loads((root / 'observer_final.json').read_text())
    outputs = []
    for reader in observation['readers']:
        p = root / Path(reader['output']).name
        raw = p.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == reader['sha256'] and reader['exit'] == 0
        assert '--nonblocking' in reader['argv'] and '--locals' not in reader['argv']
        z = json.loads(raw)
        frames = z['shared']['frames']
        main = [q for q in z['profiles'] if '"MainThread"' in q['name']]
        assert len(main) == 1
        main = main[0]
        counters = collections.Counter()
        leaf_counts = collections.Counter()
        native_contexts = collections.Counter()
        examples = {}
        bad = []
        for sample_id, stack in enumerate(main['samples']):
            fs = [frames[i] for i in stack]
            if not fs:
                bad.append(sample_id)
                continue
            names = {(f['file'], f['name']) for f in fs}
            target = any('/vllm_ascend/worker/' in file and name in ('execute_model', 'sample_tokens')
                         for file, name in names)
            counters['runner_ancestry' if target else 'outside_runner_or_truncated'] += 1
            # Family labels are mutually exclusive sampled contexts, not CPU categories.
            if any('/ops/fused_moe/' in file for file, _ in names):
                counters['MoE_context'] += 1
            elif any('/attention/' in file or '/ops/mla.py' in file for file, _ in names):
                counters['attention_context'] += 1
            elif target:
                counters['other_runner_context'] += 1
            else:
                counters['outside_runner_context'] += 1
            leaf = fs[-1]
            key = (leaf['file'], leaf['name'], leaf['line'])
            leaf_counts[key] += 1
            if leaf['file'].endswith('/torch/_ops.py'):
                # This location submits its inner op; it is not _ops.__call__ self time.
                key = tuple((f['file'], f['name'], f['line']) for f in fs[-5:])
                native_contexts[key] += 1
                examples.setdefault(key, dict(sample_index=sample_id, frame_ids=stack,
                                               stack=fs))
        match = re.search(r'Samples: (\d+) Errors: (\d+)', reader['stdout_tail'])
        assert match
        samples, errors = map(int, match.groups())
        assert samples == sum(len(q.get('samples', [])) for q in z['profiles'])
        cpu = (reader['reader_user_ticks'] + reader['reader_system_ticks']) / reader['clock_ticks_per_second']
        outputs.append(dict(worker=reader['worker'],profile_identity=dict(path=p.name,sha256=reader['sha256'],bytes=len(raw)),
            main_profile=main['name'],main_samples=len(main['samples']),counts=dict(counters),
            all_thread_samples=samples,reported_errors=errors,reader_CPU_seconds_before_stop=cpu,
            observer_elapsed_seconds=(observation['stopped_ns']-reader['reader_ready_ns'])/1e9,
            profile_weight_seconds=main['endValue']-main['startValue'],empty_sample_indices=bad,
            coverage_count_pass=counters['runner_ancestry'] >= 100,
            top_leaves=[dict(file=k[0],name=k[1],line=k[2],samples=v) for k,v in leaf_counts.most_common(25)],
            torch_op_leaf_callers=[dict(samples=v,frames=[dict(file=f,name=n,line=line) for f,n,line in k],example=examples[k])
                                  for k,v in native_contexts.most_common(8)]))
    guards = [json.loads((root / p).read_text()) for p in ('guards_before.json', 'guards_after.json')]
    assert guards[0] == guards[1]
    native = [json.loads((root / p).read_text()) for p in ('native_before.json', 'native_after.json')]
    assert native[0] == native[1] and native[0]['all_rank_original_library']
    assert json.loads((root / 'cleanup_errors.json').read_text()) == []
    result = dict(run_id='GLM-RUN-0257',execution='completed',function_context_observed=True,
        diagnostic_verdict='INCONCLUSIVE because readers report errors; frozen request budget exhausted',
        workers=outputs,no_model_retry=True,unchanged_stock_services=True,performance_claim=False,
        limits=['Nonblocking stacks can be inconsistent or truncated; errors invalidate quantitative coverage under the frozen rule.',
                'Per-thread recorded weight omits idle/failed observations and is not Decode wall duration.',
                'Readers consume about one CPU core each; no timing/cost/gain or OFF265ms decomposition comes from this diagnostic.',
                'torch._ops.__call__ leaves include the native kernel frontend; do not remove the wrapper and claim its entire sample count.',
                'No native frames or locals: use validated existing Run249/CANN and Run254 callchains for native attribution.',
                'Scheduler/Executor/Core are not observed by these worker source stacks.'])
    (root / 'source_functions_reduced.json').write_text(json.dumps(result,indent=2)+'\n')
    return {k:[{q:r[q] for q in ('main_samples','counts','reported_errors','reader_CPU_seconds_before_stop')} for r in outputs] if k=='workers' else v for k,v in result.items() if k!='limits'}


if __name__ == '__main__':
    print(json.dumps(reduce(Path(sys.argv[1])),indent=2))
