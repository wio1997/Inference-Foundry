"""Run one bounded GLM text request using an externally launched Worker group.

Each rank must be launched by the caller with exactly one visible NPU. This
example uses the experimental four-rank offline entry; it does not launch or
reserve devices and does not establish model-quality or scale acceptance.
"""
import argparse
import os
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local GLM checkpoint directory')
    parser.add_argument('--sfa-library', required=True, help='Installed Ascend extension library')
    parser.add_argument('--init-method', required=True, help='Shared distributed rendezvous URL')
    parser.add_argument('--rank', type=int, required=True, choices=range(4))
    parser.add_argument('--tokens', type=int, nargs='+', required=True, help='Prompt token IDs')
    parser.add_argument('--max-tokens', type=int, default=1, choices=range(1, 9))
    parser.add_argument('--max-context', type=int, default=32, choices=(32, 64))
    parser.add_argument('--kda-backend', default='ascendc',
        choices=('ascendc', 'ascendc_installed_decode'),
        help='Experimental KDA caller; installed decode has not passed model-quality acceptance')
    parser.add_argument('--shard-dense-down', action='store_true',
        help='Experimental rank-sharded dense down projection; quality not accepted')
    parser.add_argument('--dense-activation', default='torch_fp32',
        choices=('torch_fp32', 'vllm_native'),
        help='Experimental dense activation; native option has not passed quality acceptance')
    args = parser.parse_args()
    if len(args.tokens) + args.max_tokens > args.max_context:
        parser.error('prompt and generated tokens must fit the selected context bound')

    os.environ['DEEPINFER_USE_NEXTN_DECODE'] = '0'
    os.environ['DEEPINFER_ZERO_OVERHEAD_SCHEDULING'] = '0'
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch_npu
    from deepinfer._C import engine
    from deepinfer.engine.arg_utils import EngineArgs
    from deepinfer.engine.engine_type import EngineType
    from deepinfer.engine.llm_engine import LLMEngine
    from deepinfer.sampling_params import SamplingParams
    from deepinfer.worker.worker import Worker

    if torch_npu.npu.device_count() != 1:
        raise RuntimeError('each rank must have exactly one visible NPU')
    torch_npu.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    parallel = types.SimpleNamespace(world_size=4, context_parallel_size=4,
        data_parallel_size=1, expert_parallel_size=1, engram_parallel_size=1,
        engine_rank=0, data_parallel_rank=0)
    model = types.SimpleNamespace(architecture='Glm5NextForConditionalGeneration', model=args.model)
    worker = Worker(EngineType.NORMAL, model, parallel, types.SimpleNamespace(),
        types.SimpleNamespace(), 0, args.rank, args.init_method, args.rank == 0,
        bounded_glm_startup=True)
    if args.rank != 0:
        return
    llm = None
    try:
        worker.run_workers('initialize_bounded_glm_text', sfa_library=args.sfa_library,
            max_read_bytes=2*1024**3, max_expert_checkpoint_bytes=512*1024**2,
            max_expert_resident_bytes=512*1024**2, max_context_tokens=args.max_context,
            max_step_tokens=2, rank_cache_bytes=128*1024**2,
            max_assembled_bytes=16*1024**2, max_intermediate_rows=12288,
            max_packed_rows=8, hc_backend='ascend_installed', kda_backend=args.kda_backend,
            shard_dense_down=args.shard_dense_down, dense_activation_backend=args.dense_activation)
        config = EngineArgs(model=args.model, max_model_len=args.max_context, max_num_batched_tokens=128,
            max_num_seqs=1, context_parallel_size=4, expert_parallel_size=1,
            engram_parallel_size=1, tokens_per_block=128, chunked_prefill=True,
            disable_log_stats=True, offline_inference=True)
        llm = LLMEngine.from_bounded_glm_worker(config, worker=worker, logical_blocks=4)
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True, seq_seed=0)
        llm.add_request('offline', 'glm-bounded-example', params, np.array(args.tokens, dtype=np.int32))
        generated = []
        for _ in range(80):
            if not llm.has_unfinished_requests():
                break
            outputs = llm.step()
            for value in outputs:
                raw = int(value)
                generated.append(int(engine.RequestOutput.get_token_id(raw)))
        else:
            raise RuntimeError('bounded scheduler iteration limit exceeded')
        print('generated_token_ids:', generated, flush=True)
    finally:
        if llm is not None:
            llm.close()
        elif dist.is_initialized():
            worker.run_workers('stop')


if __name__ == '__main__':
    main()
