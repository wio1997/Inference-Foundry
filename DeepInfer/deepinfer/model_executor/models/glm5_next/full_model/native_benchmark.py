# SPDX-License-Identifier: Apache-2.0
"""Offline, single-active-request latency measurements on the native engine."""
import json
import statistics
import time


def run_native_benchmark(llm, worker, args, out):
    import numpy as np
    from deepinfer._C import engine
    from deepinfer.sampling_params import SamplingParams

    tokens = args.benchmark_tokens or args.tokens
    count = args.benchmark_output_tokens
    params = SamplingParams(temperature=0, max_tokens=count, ignore_eos=True, seq_seed=0)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "requests.jsonl").exists():
        raise FileExistsError("benchmark output already exists; inspect prior run before reusing")
    rows = []
    baseline_memory = None
    for trial in range(args.warmup_requests + args.measured_requests):
        warmup = trial < args.warmup_requests
        if trial == args.warmup_requests:
            baseline_memory = worker.run_workers('benchmark_memory_snapshot', reset=True)
        request_id = 'benchmark-' + str(trial)
        started = time.perf_counter_ns()
        llm.add_request('offline', request_id, params, np.array(tokens, dtype=np.int32))
        timestamps, generated = [], []
        finish_reason = None
        empty_steps = 0
        for _ in range(args.max_context + count + 256):
            if not llm.has_unfinished_requests():
                break
            outputs = llm.step()
            ready = time.perf_counter_ns()
            if len(outputs) > 1:
                raise RuntimeError('latency protocol requires one non-speculative token per step')
            if not len(outputs):
                empty_steps += 1
            for ptr in outputs:
                ptr = int(ptr)
                token = int(engine.RequestOutput.get_token_id(ptr))
                if token < 0:
                    raise RuntimeError('placeholder output is invalid for benchmark')
                generated.append(token); timestamps.append(ready)
                if engine.RequestOutput.get_finished(ptr):
                    finish_reason = engine.RequestOutput.get_finish_reason(ptr)
        else:
            raise RuntimeError('benchmark scheduler iteration budget exhausted')
        drained = time.perf_counter_ns()
        if len(generated) != count or worker._glm_worker_runtime.backend._owner is not None:
            raise RuntimeError('benchmark token count or terminal cache owner invalid')
        first, last = timestamps[0], timestamps[-1]
        row = {'request_id': request_id, 'warmup': warmup,
               'prompt_token_ids': tokens, 'generated_token_ids': generated,
               'started_ns': started, 'sample_ready_ns': timestamps, 'drained_ns': drained,
               'ttft_seconds': (first - started) / 1e9,
               'tpot_seconds': (last - first) / ((count - 1) * 1e9),
               'output_tokens_per_second': count * 1e9 / (last - started),
               'request_elapsed_seconds': (drained - started) / 1e9,
               'empty_scheduler_steps': empty_steps, 'finish_reason': finish_reason,
               'driver_owner_released': True}
        rows.append(row)
        # Write only after a request, outside its token timing window.
        with (out / 'requests.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps({'benchmark_request': request_id, 'warmup': warmup,
                          'ttft_seconds': row['ttft_seconds'], 'tpot_seconds': row['tpot_seconds']}), flush=True)
    memory = worker.run_workers('benchmark_memory_snapshot', reset=False)
    measured = [r for r in rows if not r['warmup']]
    summary = {'status': 'executed', 'scope': 'native offline single-active-request',
               'world_size': args.world_size, 'concurrency': 1,
               'input_tokens': len(tokens), 'output_tokens': count,
               'warmup_requests': args.warmup_requests, 'measured_requests': len(measured),
               'ignore_eos': True, 'timing_clock': 'perf_counter_ns',
               'ttft_definition': 'add_request start to first native step returned token',
               'tpot_definition': '(last token ready - first token ready)/(output_tokens-1)',
               'model_load_included': False, 'per_token_disk_logging': False,
               'engineering_guard': 'requires separate frozen-case scoring; not formal quality approval',
               'memory_before_measured': baseline_memory, 'memory_after_measured': memory}
    for key in ['ttft_seconds', 'tpot_seconds', 'output_tokens_per_second']:
        values = [r[key] for r in measured]
        summary[key] = {'median': statistics.median(values), 'mean': statistics.mean(values),
                        'min': min(values), 'max': max(values), 'samples': values}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary
