"""Native paged W8A8 complete-model startup entry; never runs on import.

Example after cards become available (requires an available target):
torchrun --nproc-per-node=16 examples/ascend910c_w8a8_packed_generate.py \
    --model-dir /models/DeepSeek-V4.1-Flash-1017-w8a8 --prompt '你好' --output result.json
"""
import argparse
import json
import os
from pathlib import Path
import time
import hashlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--prompt')
    p.add_argument('--prompt-token-ids', type=Path)
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--max-tokens', type=int, default=34816)
    p.add_argument('--chunk-size', type=int, default=64)
    p.add_argument('--tp-size', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    # CANN custom metadata and HCCL must agree on the process-local device.
    # Isolate one physical logical-device per worker BEFORE importing torch;
    # every worker then uses local npu:0, like the proven Ascend worker setup.
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(rank)
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from tokenizers import Tokenizer
    from deepinfer.engine.ascend_w8a8_packed import AscendW8A8PackedEngine
    from deepinfer.npu_ops.dsa910c import load_ops
    load_ops()
    torch.npu.set_device(0)
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        dist.init_process_group('hccl')
    tp_group = None
    if args.tp_size > 1:
        world = dist.get_world_size()
        if world % args.tp_size:
            raise ValueError('world size must divide attention TP size')
        for first in range(0, world, args.tp_size):
            group = dist.new_group(list(range(first, first+args.tp_size)), backend='hccl')
            if first <= rank < first+args.tp_size:
                tp_group = group
    tokenizer = Tokenizer.from_file(str(args.model_dir/'tokenizer.json'))
    config = json.loads((args.model_dir/'config.json').read_text())
    if args.prompt_token_ids:
        prompt_ids = json.loads(args.prompt_token_ids.read_text())
    elif args.prompt is not None:
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
        if not prompt_ids or prompt_ids[0] != config['bos_token_id']:
            prompt_ids.insert(0, config['bos_token_id'])
    else:
        raise ValueError('provide --prompt or --prompt-token-ids')
    if len(prompt_ids)+args.max_new_tokens > args.max_tokens:
        raise ValueError('prompt and requested generation exceed context limit')
    start = time.monotonic()
    progress = lambda event: print(json.dumps({'rank': rank, **event}), flush=True)
    engine = AscendW8A8PackedEngine(args.model_dir, device='npu:0', max_tokens=args.max_tokens, chunk_size=args.chunk_size, tp_group=tp_group, progress=progress)
    loaded = time.monotonic()
    generated = engine.generate_ids(prompt_ids, max_new_tokens=args.max_new_tokens, eos_token_id=config['eos_token_id'])
    torch.npu.synchronize()
    if rank == 0:
        result = {'checkpoint': str(args.model_dir), 'backend': 'DeepInfer native paged Ascend W8A8/INT8',
                  'prompt': args.prompt, 'input_tokens': len(prompt_ids),
                  'prompt_token_ids': prompt_ids if len(prompt_ids) <= 512 else None,
                  'prompt_token_ids_file': str(args.prompt_token_ids) if args.prompt_token_ids else None,
                  'prompt_token_ids_sha256': hashlib.sha256(json.dumps(prompt_ids,separators=(',',':')).encode()).hexdigest(),
                  'generated_token_ids': generated, 'runtime_stats': engine.model.runtime_stats(),
                  'text': tokenizer.decode(generated, skip_special_tokens=False),
                  'load_seconds': loaded-start, 'generation_seconds': time.monotonic()-loaded,
                  'resident_logical_bytes': engine.model.resident_bytes(),
                  'peak_npu_allocated_bytes': torch.npu.max_memory_allocated(),
                  'context_limit': args.max_tokens, 'engram': 'CPU selected-row offload',
                  'attention_tp_size': args.tp_size, 'expert_parallel_size': dist.get_world_size(),
                  'short_prefix_attention': 'native eager while all compressed keys fit top512; CANN for indexed long context',
                  'scope': 'native paged full40-layer generation; TP attention and global EP; not final serving/performance'}
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
