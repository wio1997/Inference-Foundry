# SPDX-License-Identifier: Apache-2.0
"""Offline native packed W8A8 entry; serving/scheduler integration follows."""
from pathlib import Path

import torch
import torch.distributed as dist

from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.ascend_w8a8_packed import AscendW8A8PackedModel


class AscendW8A8PackedEngine:
    def __init__(self, model_dir, *, device, max_tokens=34816, requests=1, chunk_size=64, group=None, tp_group=None, progress=None):
        self.group = group
        reader = ModelSlimCheckpoint(Path(model_dir))
        self.model = AscendW8A8PackedModel(reader, device, max_tokens=max_tokens,
            requests=requests, chunk_size=chunk_size, group=group, tp_group=tp_group, progress=progress).eval()
        self.progress = progress or (lambda event: None)

    @torch.inference_mode()
    def generate_batch(self, prompts, *, max_new_tokens=32, eos_token_id=None):
        prompts = [list(ids) for ids in prompts]
        if (not prompts or len(prompts) > self.model.requests or max_new_tokens < 1
                or any(not ids or len(ids)+max_new_tokens > self.model.max_tokens for ids in prompts)):
            raise ValueError('prompts and output length must fit the native cache capacity')
        self.model.reset()
        logits = []
        for slot, prompt in enumerate(prompts):
            for offset in range(0, len(prompt), self.model.chunk_size):
                current = self.model.forward_tokens([prompt[offset:offset+self.model.chunk_size]], [slot])
                if rank_is_root(self.group):
                    self.progress({'stage': 'prefill', 'request_slot': slot,
                                   'computed_tokens': self.model.next_positions[slot], 'input_tokens': len(prompt)})
            logits.append(current)
        logits = torch.cat(logits)
        active, generated = list(range(len(prompts))), [[] for _ in prompts]
        rank = dist.get_rank(self.group) if dist.is_initialized() else 0
        root = dist.get_global_rank(self.group, 0) if self.group is not None else 0
        for step in range(max_new_tokens):
            tokens = logits.argmax(-1).long() if rank == 0 else torch.empty(len(active), dtype=torch.int64, device=logits.device)
            if dist.is_initialized():
                dist.broadcast(tokens, root, group=self.group)
            values = tokens.cpu().tolist()
            next_slots, next_chunks = [], []
            for slot, value in zip(active, values):
                generated[slot].append(value)
                if value != eos_token_id:
                    next_slots.append(slot); next_chunks.append([value])
            if not next_slots or step+1 == max_new_tokens:
                break
            logits = self.model.forward_tokens(next_chunks, next_slots)
            active = next_slots
        return generated

    def generate_ids(self, prompt_ids, **kwargs):
        return self.generate_batch([prompt_ids], **kwargs)[0]


def rank_is_root(group):
    return not dist.is_initialized() or dist.get_rank(group) == 0
