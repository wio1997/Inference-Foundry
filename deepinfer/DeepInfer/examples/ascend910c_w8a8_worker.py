"""Existing W8A8 loader with the normal original Engine/Worker service entry."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source))
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--control-dir', type=Path, required=True)
    p.add_argument('--tp-size', type=int, default=8)
    p.add_argument('--slots-per-dp', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=34816)
    p.add_argument('--max-model-len', type=int)
    p.add_argument('--chunk-size', type=int, default=4096)
    p.add_argument('--prefix-tokens', type=int, default=29568)
    p.add_argument('--mc2-buffer-mib', type=int, default=2560)
    args = p.parse_args()
    rank = int(os.environ['RANK'])
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(int(os.environ['LOCAL_RANK']))
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['ASCEND_RT_VISIBLE_DEVICES']
    os.environ.setdefault('DEEPINFER_910C_SOFTWARE_OPS', '1')
    if os.environ.get('DEEPINFER_SINGLE_POINT_ABLATION') == '1':
        raise ValueError('original Engine variants require separate immutable startup configurations')
    import deepinfer
    if not Path(deepinfer.__file__).resolve().is_relative_to(source):
        raise RuntimeError('original Engine source was shadowed')
    from deepinfer import envs
    from deepinfer.engine.arg_utils import EngineArgs
    from deepinfer.config import _get_runtime_max_model_len
    from deepinfer.entrypoints.existing_w8a8 import attach_original_engine
    from deepinfer.entrypoints.engine_job import run_engine_job
    from deepinfer.entrypoints.file_transport import atomic_json, heartbeat, read_ready, source_identity
    from deepinfer.model_executor.models.w8a8_execution910c import OriginalW8A8Executor
    from deepinfer.model_executor.models.hidden_graph_layout910c import validate_hidden_capacity
    from deepinfer.engine.ascend_w8a8_serving import NativeServingRunner
    import torch
    import torch.distributed as dist
    import torch_npu
    from deepinfer.npu_ops.dsa910c import load_ops
    load_ops()
    torch.npu.set_device(0)
    dist.init_process_group('hccl')
    world = dist.get_world_size()
    if args.tp_size < 1 or world % args.tp_size:
        raise ValueError('context parallel size must divide the world')
    dp_size = world // args.tp_size
    maximum = args.max_tokens - (_get_runtime_max_model_len(1)-1)
    engine_args = EngineArgs(model=args.model_dir, served_model_name='dsv41',
        data_parallel_size=dp_size, context_parallel_size=args.tp_size,
        expert_parallel_size=world, engram_parallel_size=world, engine_rank=rank//args.tp_size,
        max_model_len=maximum if args.max_model_len is None else args.max_model_len,
        max_num_seqs=args.slots_per_dp, max_num_batched_tokens=args.chunk_size,
        chunked_prefill=True, disable_log_stats=True)
    # Resolve original config/data imports before reading any weight tensor.
    config, _, _, _ = engine_args.create_engine_configs()
    if config.runtime_max_model_len > args.max_tokens:
        raise ValueError('runtime positions exceed the requested resident cache capacity')
    if envs.DEEPINFER_USE_NEXTN_DECODE < 0:
        raise ValueError('original nextn must be nonnegative')
    if not envs.DEEPINFER_ENFORCE_EAGER:
        specs = config.model_cfg.gpt.attn_module_list
        sources = {source for spec in specs if spec is not None
                   for source in spec.cfg.get('layer_info_shared_cfg', {}).get('layer_kv_map', {}).values()}
        has_c2 = any(specs[source].cfg.get('kv_compress_ratio') == 2 for source in sources)
        validate_hidden_capacity(args.slots_per_dp * (1 + envs.DEEPINFER_USE_NEXTN_DECODE),
            args.max_tokens, c2_logical_rows=(args.max_tokens + 1)//2 if has_c2 else None)
    os.environ['DEEPINFER_910C_DSPARK'] = '1' if envs.DEEPINFER_USE_NEXTN_DECODE else '0'
    if any(args.control_dir.glob('ready-rank*.json')):
        raise ValueError('use a fresh control directory for the new Engine deployment')
    tp_group = None
    for first in range(0, world, args.tp_size):
        group = dist.new_group(list(range(first, first+args.tp_size)), backend='hccl')
        if first <= rank < first+args.tp_size:
            tp_group = group
    if os.environ.get('DEEPINFER_910C_MC2') == '1':
        from deepinfer.npu_ops.mc2_910c import initialize_mc2_group
        initialize_mc2_group(args.mc2_buffer_mib)
    progress = lambda event: print(json.dumps(dict(rank=rank, **event)), flush=True)
    runner = NativeServingRunner(args.model_dir, tp_group=tp_group,
        slots_per_dp=args.slots_per_dp, prefix_tokens=args.prefix_tokens,
        device='npu:0', max_tokens=args.max_tokens, chunk_size=args.chunk_size, progress=progress)
    backend = OriginalW8A8Executor(runner)
    # Followers enter original Worker RPC; drivers retain the original Engine.
    engine = attach_original_engine(backend, engine_args)
    engine._run_workers('publish_backend_ready', control_dir=str(args.control_dir))
    peers = read_ready(args.control_dir, world_size=world, source=source_identity())
    previous, last_heartbeat = None, 0.
    def alive(job_id=None):
        nonlocal last_heartbeat
        now = time.monotonic()
        if now-last_heartbeat >= .5:
            heartbeat(args.control_dir, engine.dp_rank, peers, job_id=job_id)
            last_heartbeat = now
    try:
        while not (args.control_dir/'shutdown.json').exists():
            if any(args.control_dir.glob('fatal-*.json')):
                raise RuntimeError('deployment has a recorded fatal error')
            alive()
            current = args.control_dir/'current.json'
            command = json.loads(current.read_text()) if current.exists() else None
            if command is None or command['id'] == previous:
                time.sleep(.005)
                continue
            previous = command['id']
            directory = args.control_dir/'jobs'/previous
            body = json.loads((directory/'request.json').read_text())
            run_engine_job(engine, body, directory, progress=lambda: alive(previous))
            progress(dict(stage='original_job_complete', job_id=previous,
                runtime_stats=runner.model.runtime_stats(),
                hidden_graph_replays={} if backend.hidden_graphs is None else
                    {str(size): frame.replays for size, frame in backend.hidden_graphs.frames.items()},
                draft_proposals=0 if runner.model.dspark is None else runner.model.dspark.proposals,
                draft_graph_replays=0 if backend.mtp_module is None or backend.mtp_module.decode_graph_runner is None
                    else backend.mtp_module.decode_graph_runner.replays))
    except Exception:
        atomic_json(args.control_dir/f'fatal-rank{rank}.json', dict(rank=rank,error=traceback.format_exc()))
        raise
    finally:
        engine.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Keep failures visible and stop the deployment instead of leaving
        # old ready files usable for a subsequent request.
        control = next((sys.argv[i+1] for i,x in enumerate(sys.argv[:-1]) if x == '--control-dir'), None)
        if control is not None:
            directory = Path(control)
            directory.mkdir(parents=True, exist_ok=True)
            rank = os.environ.get('RANK', 'unknown')
            path = directory/f'fatal-rank{rank}.json'
            path.write_text(json.dumps(dict(error=traceback.format_exc(), rank=rank)))
        raise
