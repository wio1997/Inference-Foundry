# SPDX-License-Identifier: Apache-2.0
"""One real DeepInfer scheduler request through the complete reference factory.

External authorized launcher owns devices, rendezvous, resource registration and
the go file. This program uses the existing bounded four-rank Worker entry. It
does not launch/reserve devices; validate every deployment against its source manifest.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import time
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--world-size', type=int, default=4, choices=(4, 8, 16))
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--local-device', type=int, required=True)
    p.add_argument('--init-method', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--go-file', type=Path, required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--tokens', type=int, nargs='+')
    inputs.add_argument('--cases', type=Path)
    inputs.add_argument('--serve', action='store_true')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=18080)
    p.add_argument('--served-model-name', default='glm53flash')
    p.add_argument('--serve-max-requests', type=int, default=0,
                   help='zero serves until launcher deadline; positive bounds integration runs')
    p.add_argument('--max-new-tokens', type=int, default=3)
    p.add_argument('--num-requests', type=int, default=1, choices=(1, 2, 3))
    p.add_argument('--max-context', type=int, default=128)
    p.add_argument('--max-step', type=int, default=128)
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--benchmark-tokens', type=int, nargs='+')
    p.add_argument('--benchmark-output-tokens', type=int, default=64)
    p.add_argument('--warmup-requests', type=int, default=1)
    p.add_argument('--measured-requests', type=int, default=3)
    p.add_argument('--weight-offload', choices=('prefetch', 'none'), default='prefetch')
    p.add_argument('--compute-backend', choices=('reference', 'sdk-batch'), default='reference')
    p.add_argument('--max-sequences', type=int, default=1)
    p.add_argument('--cache-bytes', type=int, default=2147483648)
    p.add_argument('--batch-smoke-cases', type=Path)
    p.add_argument('--graph-decode', action='store_true')
    a = p.parse_args()
    if a.graph_decode and a.compute_backend != 'sdk-batch':
        p.error('graph decode requires SDK compute backend')
    if not 1 <= a.max_sequences <= 8 or (a.compute_backend == 'reference' and a.max_sequences != 1):
        p.error('explicit SDK batch backend required for multi-sequence admission')
    if (a.compute_backend == 'sdk-batch' and (
            not (a.serve or a.batch_smoke_cases is not None) or a.benchmark or a.cases is not None)):
        p.error('SDK batch backend currently supports explicit serve or batch-smoke entry only')
    if not 1 <= a.port <= 65535 or a.serve_max_requests < 0:
        p.error('valid port and nonnegative service request bound required')
    if (not a.output_dir.is_absolute() or not a.go_file.is_absolute()
            or not 0 <= a.rank < a.world_size
            or not 1 <= a.max_new_tokens <= 512
            or (a.tokens is not None and len(a.tokens) + a.max_new_tokens > a.max_context)
            or (a.cases is not None and (not a.cases.is_absolute() or not a.cases.is_file()))
            or not 1 <= a.max_step <= a.max_context):
        p.error('absolute task output/go paths and positive bounded token sizes required')
    if a.benchmark and (not (a.benchmark_tokens or a.tokens)
            or not 2 <= a.benchmark_output_tokens <= 512
            or not 1 <= a.warmup_requests <= 8 or not 1 <= a.measured_requests <= 32
            or len(a.benchmark_tokens or a.tokens) + a.benchmark_output_tokens > a.max_context):
        p.error('benchmark requires explicit tokens, fitting context, warmup and bounded repetitions')
    if a.compute_backend == 'sdk-batch':
        # DP1 online admission should inspect waiting work each iteration,
        # rather than inheriting the legacy 70/100 decode-step polling cadence.
        os.environ['DEEPINFER_PREFILL_ATTEMPT_INTERVAL'] = '1'
        os.environ['DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL'] = '1'
    os.environ['DEEPINFER_USE_NEXTN_DECODE'] = '0'
    os.environ['DEEPINFER_ZERO_OVERHEAD_SCHEDULING'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    out = a.output_dir / ('rank' + str(a.rank))
    out.mkdir(parents=True, exist_ok=True)
    for key in ('HF_HOME', 'XDG_CACHE_HOME', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR'):
        target = out / 'cache' / key.lower()
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    import numpy as np
    import torch
    import torch_npu
    import torch.distributed as dist
    if not 0 <= a.local_device < torch_npu.npu.device_count():
        raise ValueError('explicit container device index outside visible devices')
    torch_npu.npu.set_device(a.local_device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.ones(1, device='npu')
    torch_npu.npu.synchronize()
    from deepinfer._C import engine
    from deepinfer.engine.arg_utils import EngineArgs
    from deepinfer.engine.engine_type import EngineType
    from deepinfer.engine.llm_engine import LLMEngine
    from deepinfer.model_executor.parallel_utils.parallel_state import PG
    from deepinfer.sampling_params import SamplingParams
    from deepinfer.worker.worker import Worker
    from deepinfer.model_executor.models.glm5_next.full_model.reference_factory import prepare_reference_worker_runtime
    from vllm_ascend.sample.sampler import AscendSampler

    (out / 'ready.json').write_text(json.dumps({'container_pid': os.getpid(), 'rank': a.rank, 'local_device': a.local_device}))
    deadline = time.monotonic() + 180
    while not a.go_file.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError('launcher did not approve host PID/device mapping')
        time.sleep(0.1)

    def initialize_reference_glm(self, **kwargs):
        if a.compute_backend == 'sdk-batch':
            from deepinfer.model_executor.models.glm5_next.full_model.sdk_runtime import prepare_sdk_compute
            runtime = prepare_sdk_compute(a.model, device=self.device,
                native_group=PG.context_parallel_group(), local_rank=a.local_device,
                init_method=a.init_method, max_context=a.max_context,
                max_step=a.max_step, max_requests=a.max_sequences, cache_bytes=a.cache_bytes, graph_decode=a.graph_decode)
            runtime.install(self)
            model = runtime.compute_worker.get_model()
            (out / 'loaded.json').write_text(json.dumps({'rank': a.rank,
                'backend': 'SDK compute worker under native scheduler;no SDK engine/scheduler',
                'max_sequences': a.max_sequences, 'cache_bytes': a.cache_bytes,
                'enforce_eager': runtime.config.model_config.enforce_eager,
                'cudagraph_mode': str(runtime.config.compilation_config.cudagraph_mode),
                'capture_sizes': runtime.config.compilation_config.cudagraph_capture_sizes,
                'parameter_numel_by_device': {d: sum(p.numel() for p in model.parameters() if p.device.type == d)
                    for d in {p.device.type for p in model.parameters()}},
                'state': 'loaded;native batch execution unverified'})+'\n')
            return
        runtime = prepare_reference_worker_runtime(
            a.model, self.device, worker_group=PG.context_parallel_group(),
            local_rank=a.local_device, sampler=AscendSampler() if self.is_driver_worker else None,
            max_context_tokens=a.max_context, max_step_tokens=a.max_step, weight_offload=a.weight_offload)
        runtime.install(self)
        (out / 'loaded.json').write_text(json.dumps({
            'rank': a.rank, 'cache_layers': runtime.cache.roles,
            'weight_offload': a.weight_offload,
            'offloader_type': type(runtime.loaded.offloader).__name__,
            'parameter_numel_by_device': {device: sum(p.numel() for p in runtime.loaded.root_model.parameters() if p.device.type == device) for device in {p.device.type for p in runtime.loaded.root_model.parameters()}},
            'source_identity': 'reference-runtime-pins.json',
            'state': 'loaded; native request execution not yet established'}, indent=2) + '\n')

    # Reuse Worker's own collective RPC and native worker loop. The only new
    # RPC constructs/installs our model backend; it never invokes vLLM schedule.
    def benchmark_memory_snapshot(self, reset=False):
        if reset:
            torch_npu.npu.reset_peak_memory_stats(self.device)
        stats = [torch_npu.npu.memory_allocated(self.device),
                 torch_npu.npu.memory_reserved(self.device),
                 torch_npu.npu.max_memory_allocated(self.device),
                 torch_npu.npu.max_memory_reserved(self.device)]
        group = PG.context_parallel_group()
        value = torch.tensor(stats, dtype=torch.int64, device=self.device)
        gathered = [torch.empty_like(value) for _ in range(group.size())]
        dist.all_gather(gathered, value, group=group)
        return [{'rank': i, 'allocated_bytes': int(v[0]), 'reserved_bytes': int(v[1]),
                 'peak_allocated_bytes': int(v[2]), 'peak_reserved_bytes': int(v[3])}
                for i, v in enumerate(torch.stack(gathered).cpu().tolist())]

    def flush_sdk_finished(self):
        self._glm_worker_runtime.adapter.flush_finished()

    Worker.flush_sdk_finished = flush_sdk_finished
    Worker.benchmark_memory_snapshot = benchmark_memory_snapshot
    Worker.initialize_reference_glm = initialize_reference_glm
    parallel = SimpleNamespace(world_size=a.world_size, context_parallel_size=a.world_size,
        data_parallel_size=1, expert_parallel_size=1, engram_parallel_size=1,
        engine_rank=0, data_parallel_rank=0)
    model = SimpleNamespace(architecture='Glm5NextForConditionalGeneration', model=a.model)
    worker = Worker(EngineType.NORMAL, model, parallel, SimpleNamespace(),
                    SimpleNamespace(), a.local_device, a.rank, a.init_method, a.rank == 0,
                    bounded_glm_startup=True)
    if a.rank != 0:
        return
    llm = None
    try:
        worker.run_workers('initialize_reference_glm')
        config = EngineArgs(model=a.model, max_model_len=a.max_context,
            max_num_batched_tokens=max(128, a.max_step), max_num_seqs=a.max_sequences,
            context_parallel_size=a.world_size, expert_parallel_size=1, engram_parallel_size=1,
            tokens_per_block=128, chunked_prefill=True,
            disable_log_stats=True, offline_inference=True)
        logical_blocks = a.max_sequences * (3 + (a.max_context - 1) // 128)
        llm = LLMEngine.from_bounded_glm_worker(config, worker=worker, logical_blocks=logical_blocks)
        if a.batch_smoke_cases is not None:
            from deepinfer.model_executor.models.glm5_next.full_model.sdk_smoke import run_batch_smoke
            run_batch_smoke(llm, worker, a, out)
            return
        if a.serve:
            if a.compute_backend == 'sdk-batch':
                from deepinfer.model_executor.models.glm5_next.full_model.sdk_http import serve_sdk_batch_engine
                serve_sdk_batch_engine(llm, worker, a, out)
                return
            from deepinfer.model_executor.models.glm5_next.full_model.native_http import serve_native_engine
            serve_native_engine(llm, worker, a, out)
            return
        if a.cases is not None:
            from deepinfer.model_executor.models.glm5_next.full_model.engineering_batch import run_engineering_batch
            quality_out = out / 'engineering' if a.benchmark else out
            quality_out.mkdir(parents=True, exist_ok=True)
            run_engineering_batch(llm, worker, a, quality_out)
            if not a.benchmark:
                return
        if a.benchmark:
            from deepinfer.model_executor.models.glm5_next.full_model.native_benchmark import run_native_benchmark
            measured = run_native_benchmark(llm, worker, a, out / 'benchmark')
            (out / 'result.json').write_text(json.dumps({
                'status': 'executed', 'scope': 'native benchmark; engineering scoring separate',
                'worker_ranks': a.world_size, 'driver_owner_released': True,
                'benchmark': measured}, indent=2) + '\n')
            return
        params = SamplingParams(temperature=0, max_tokens=a.max_new_tokens, ignore_eos=True, seq_seed=0)
        request_results = []
        for trial in range(a.num_requests):
            request_id = 'reference-fullmodel-' + str(trial + 1)
            llm.add_request('offline', request_id, params, np.array(a.tokens, dtype=np.int32))
            generated = []
            for _ in range(a.max_context + a.max_new_tokens + 8):
                if not llm.has_unfinished_requests():
                    break
                for value in llm.step():
                    generated.append(int(engine.RequestOutput.get_token_id(int(value))))
                    progress = {'request_id': request_id, 'generated_token_ids': generated,
                                'status': 'partial_unverified', 'scope': 'native scheduler outputs observed so far'}
                    (out / 'progress.json').write_text(json.dumps(progress) + '\n')
                    print(json.dumps(progress), flush=True)
            else:
                raise RuntimeError('bounded native scheduler iteration limit exceeded')
            if len(generated) != a.max_new_tokens:
                raise ValueError('native request did not produce its explicit generation cap')
            if worker._glm_worker_runtime.backend._owner is not None:
                raise ValueError('terminal native request retained its rank-zero cache owner')
            request_results.append({'request_id': request_id, 'generated_token_ids': generated,
                                    'driver_owner_released': True})
            (out / 'completed-requests.json').write_text(json.dumps(request_results) + '\n')
        (out / 'result.json').write_text(json.dumps({
            'status': 'executed', 'generated_token_ids': generated,
            'requests': request_results,
            'same_prompt_outputs_equal': all(x['generated_token_ids'] == request_results[0]['generated_token_ids'] for x in request_results),
            'prompt_token_ids': a.tokens, 'scope': 'sequential native full-model requests; quality unscored',
            'max_context_tokens': a.max_context, 'max_step_tokens': a.max_step,
            'driver_owner_released': True,
            'worker_ranks': a.world_size, 'physical_card_acceptance': 'not established'}, indent=2) + '\n')
        print(json.dumps({'generated_token_ids': generated}), flush=True)
    finally:
        if llm is not None:
            try:
                if (a.compute_backend == 'sdk-batch' and sys.exc_info()[0] is None
                        and not llm.has_unfinished_requests()):
                    worker.run_workers('flush_sdk_finished')
            finally:
                llm.close()
        elif dist.is_initialized():
            worker.run_workers('stop')


if __name__ == '__main__':
    main()
