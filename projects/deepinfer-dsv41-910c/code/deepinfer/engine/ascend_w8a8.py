# SPDX-License-Identifier: Apache-2.0
"""Explicit native 910C W8A8 engine entry, independent of the 950 scheduler.

Importing this module is CPU-safe. Construction is an explicit NPU operation;
do not construct it while the user's current no-card constraint is in force.
"""
from pathlib import Path

import torch
import torch.distributed as dist

from deepinfer.hugging_face.config import load_hf_model_config
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.ascend_w8a8 import AscendW8A8Model, validate_short_context


class AscendW8A8Engine:
    def __init__(self, model_dir, *, device='npu:0', max_tokens=512, group=None, progress=None):
        model_dir = Path(model_dir)
        cfg = load_hf_model_config(model_dir/'config.json')
        if not cfg.gpt.ascend_w8a8:
            raise ValueError('AscendW8A8Engine requires quant_method=ascend')
        self.reader = ModelSlimCheckpoint(model_dir)
        validate_short_context(self.reader.config, max_tokens)
        if not str(device).startswith('npu:'):
            raise ValueError('native W8A8 engine requires an explicit NPU device')
        # Only this explicit execution boundary registers/initializes NPU code.
        import torch_npu  # noqa: F401
        self.group = group
        self.model = AscendW8A8Model(self.reader, device, max_tokens=max_tokens, group=group, progress=progress).eval()

    @torch.inference_mode()
    def generate_ids(self, prompt_ids, *, max_new_tokens=32, eos_token_id=None):
        prompt_ids = list(prompt_ids)
        if not prompt_ids or max_new_tokens < 1 or len(prompt_ids)+max_new_tokens > self.model.max_tokens:
            raise ValueError('nonempty prompt and positive generation length must fit context limit')
        self.model.reset()
        for token_id in prompt_ids:
            logits = self.model.step(int(token_id))
        generated = []
        rank = dist.get_rank(self.group) if dist.is_initialized() else 0
        root_rank = dist.get_global_rank(self.group, 0) if self.group is not None else 0
        for index in range(max_new_tokens):
            token = logits.argmax(-1).to(torch.int64) if rank == 0 else torch.empty(1, dtype=torch.int64, device=logits.device)
            if dist.is_initialized():
                dist.broadcast(token, root_rank, group=self.group)
            token_id = int(token.item())
            generated.append(token_id)
            if token_id == eos_token_id or index+1 == max_new_tokens:
                break
            logits = self.model.step(token_id)
        return generated
