"""Explicit NPU entry for future W8A8 engine validation; never runs on import.

Example after cards become available (not authorized for execution now):
torchrun --nproc-per-node=16 examples/ascend910c_w8a8_generate.py \
    --model-dir /models/DeepSeek-V4.1-Flash-1017-w8a8 --prompt '你好' --output result.json
"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from tokenizers import Tokenizer
    from deepinfer.engine.ascend_w8a8 import AscendW8A8Engine
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.npu.set_device(rank)
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        dist.init_process_group('hccl')
    tokenizer = Tokenizer.from_file(str(args.model_dir/'tokenizer.json'))
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    config = json.loads((args.model_dir/'config.json').read_text())
    if not prompt_ids or prompt_ids[0] != config['bos_token_id']:
        prompt_ids.insert(0, config['bos_token_id'])
    if len(prompt_ids)+args.max_new_tokens > args.max_tokens:
        raise ValueError('prompt and requested generation exceed context limit')
    start = time.monotonic()
    progress = lambda event: print(json.dumps({'rank': rank, **event}), flush=True)
    engine = AscendW8A8Engine(args.model_dir, device=f'npu:{rank}', max_tokens=args.max_tokens, progress=progress)
    loaded = time.monotonic()
    generated = engine.generate_ids(prompt_ids, max_new_tokens=args.max_new_tokens, eos_token_id=config['eos_token_id'])
    torch.npu.synchronize()
    if rank == 0:
        result = {'checkpoint': str(args.model_dir), 'backend': 'DeepInfer native Ascend W8A8/INT8',
                  'prompt': args.prompt, 'prompt_token_ids': prompt_ids, 'generated_token_ids': generated,
                  'text': tokenizer.decode(generated, skip_special_tokens=False),
                  'load_seconds': loaded-start, 'generation_seconds': time.monotonic()-loaded,
                  'resident_logical_bytes': engine.model.resident_bytes(),
                  'peak_npu_allocated_bytes': torch.npu.max_memory_allocated(),
                  'context_limit': args.max_tokens, 'engram': 'CPU selected-row offload',
                  'scope': 'single text sequence; greedy; all backbone layers; no MTP/vision/long-context indexer'}
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
