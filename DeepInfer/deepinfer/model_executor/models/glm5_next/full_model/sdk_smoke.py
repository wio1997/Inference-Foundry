"""Bounded real-NPU native-scheduler batch and cache-lifecycle check."""
import json
from pathlib import Path


def run_batch_smoke(llm, worker, args, out):
    import numpy as np
    from deepinfer._C import engine
    from deepinfer.sampling_params import SamplingParams
    from deepinfer.model_executor.models.glm5_next.scheduler_eos import configure_scheduler_eos

    runtime = worker._glm_worker_runtime
    tokenizer = llm.tokenizer
    generation = json.loads((Path(args.model) / 'generation_config.json').read_text())
    configure_scheduler_eos(llm.scheduler, generation,
        tokenizer_eos_id=tokenizer.eos_token_id, vocab_size=runtime.vocab_size)
    rows = [json.loads(line) for line in args.batch_smoke_cases.read_text().splitlines() if line.strip()][:2]
    if len(rows) != 2 or args.max_sequences < 2:
        raise ValueError('two frozen cases and actual capacity >=2 required')
    results = []
    # First two requests are admitted together. The third tests released-slot reuse.
    for wave, selected in enumerate((rows, rows[:1])):
        active = {}
        for index, row in enumerate(selected):
            rid = f'batch-smoke-{wave}-{index}'
            params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens,
                                    ignore_eos=False, seq_seed=0)
            seq = llm.add_request('smoke', rid, params, np.asarray(row['token_ids'], dtype=np.int32))
            if seq is None:
                raise ValueError('native scheduler rejected frozen smoke request')
            active[rid] = {'request_id': rid, 'case_id': row['case_id'], 'tokens': []}
        for step in range(args.max_context + args.max_new_tokens + 64):
            if not llm.has_unfinished_requests():
                break
            for ptr in llm.step():
                ptr = int(ptr)
                rid = engine.RequestOutput.get_request_id(ptr)
                row = active[rid]
                token = int(engine.RequestOutput.get_token_id(ptr))
                if token >= 0:
                    row['tokens'].append(token)
                if engine.RequestOutput.get_finished(ptr):
                    row['finish_reason'] = engine.RequestOutput.get_finish_reason(ptr)
            (out / 'progress.json').write_text(json.dumps({'wave': wave, 'step': step,
                'tokens': {rid: len(row['tokens']) for rid, row in active.items()}}))
        else:
            raise RuntimeError('native batch smoke exceeded iteration bound')
        if runtime.adapter.controller.requests:
            raise RuntimeError('terminal native requests retained SDK allocator ownership')
        for row in active.values():
            if not row['tokens'] or 'finish_reason' not in row:
                raise RuntimeError('native request lacks generated output or terminal event')
            row['raw_text'] = tokenizer.decode(row['tokens'], skip_special_tokens=False)
            results.append(row)
        (out / 'smoke-outputs.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    sizes = runtime.adapter.batch_sizes
    if max(sizes, default=0) < 2:
        raise RuntimeError('HTTP queuing or serial dispatch is not native batch execution')
    result = {'status': 'executed', 'scope': 'native two-request batch plus slot reuse;quality unscored',
        'worker_ranks': args.world_size, 'max_actual_batch': max(sizes),
        'batch_sizes': sizes, 'requests': results, 'cache_owners_after_completion': 0}
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
