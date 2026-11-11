"""Avoid rebuilding raw vocabulary rows for native MTP output probabilities.

The pinned sampler gathers copies for both bonus/target processing and leaves
the original raw logits unchanged. For raw logprobs, those original rows are
therefore the source that the generic implementation reconstructs by scatter.
This instance-only adapter preserves real token logprobs and token ranks. It
does not change sampling, acceptance, model math or the request parameters.
"""
import hashlib
import inspect
from pathlib import Path
from types import MethodType


class NativeMTPLogprobGather:
    def __init__(self, runner):
        import torch
        from vllm.v1.sample.rejection_sampler import RejectionSampler
        from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
        sampler = runner.rejection_sampler
        if type(sampler) is not AscendRejectionSampler or sampler.is_processed_logprobs_mode:
            raise ValueError('native raw-row gather requires the pinned raw-mode Ascend sampler')
        pins = {
            RejectionSampler: '0bb9d224887ccb1355e16199005cc23a4afc6fa75156eca419851b7247d99750',
            AscendRejectionSampler: 'bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda',
        }
        for cls, expected in pins.items():
            if hashlib.sha256(Path(inspect.getsourcefile(cls)).read_bytes()).hexdigest() != expected:
                raise ValueError('raw-logit ownership source contract changed')
        if '_get_logprobs_tensors' in sampler.__dict__:
            raise ValueError('rejection sampler instance already has an owner')
        self.sampler = sampler
        self.original = sampler._get_logprobs_tensors
        self.calls = self.fallback_calls = self.validated_calls = 0
        self.validation_shapes = []
        self.stats_path = None
        self.closed = False
        owner = self

        def gather(instance, max_num_logprobs, metadata, logits, target_logits, bonus_logits, sampled_token_ids):
            if max_num_logprobs != 0 or instance.is_processed_logprobs_mode:
                owner.fallback_calls += 1
                return owner.original(max_num_logprobs, metadata, logits, target_logits, bonus_logits, sampled_token_ids)
            owner.calls += 1
            # Same row/token mapping as the pinned method; no rejected-token
            # compaction or host inspection of live acceptance state.
            starts = torch.zeros_like(metadata.cu_num_sampled_tokens)
            starts[1:] = metadata.cu_num_sampled_tokens[:-1]
            offsets = torch.arange(sampled_token_ids.shape[-1], device=starts.device, dtype=starts.dtype)
            rows = (starts.unsqueeze(1) + offsets.unsqueeze(0)).flatten()
            rows.clamp_(max=logits.shape[0] - 1)
            tokens = sampled_token_ids.clone().flatten()
            tokens.masked_fill_(tokens == -1, 0)
            selected = torch.index_select(logits.to(torch.float32), 0, rows.to(torch.int64))
            probabilities = selected if instance.is_logits_logprobs_mode else instance.sampler.compute_logprobs(selected)
            result = instance.sampler.gather_logprobs(probabilities, max_num_logprobs, tokens.to(torch.int64))
            # Three real NPU calls before formal scoring prove the retained
            # token/logprob/rank contract against the original implementation.
            if owner.validated_calls < 3:
                reference = owner.original(max_num_logprobs, metadata, logits, target_logits, bonus_logits, sampled_token_ids)
                if not all(torch.equal(getattr(result, key), getattr(reference, key)) for key in
                           ('logprob_token_ids', 'logprobs', 'selected_token_ranks')):
                    raise ValueError('native raw-row gather differs from original NPU logprobs')
                owner.validation_shapes.append({'logits': list(logits.shape), 'sampled': list(sampled_token_ids.shape)})
                owner.validated_calls += 1
                if owner.stats_path is not None:
                    import json
                    owner.stats_path.write_text(json.dumps(owner.snapshot()) + '\n')
            return result

        self.installed = MethodType(gather, sampler)
        sampler._get_logprobs_tensors = self.installed

    def snapshot(self):
        return {'enabled': True, 'closed': self.closed, 'calls': self.calls,
                'fallback_calls': self.fallback_calls, 'validated_calls': self.validated_calls,
                'validation_shapes': self.validation_shapes,
                'contract': 'real raw token logprobs and ranks;sampling unchanged;no full-vocabulary scatter reconstruction'}

    def close(self):
        if self.sampler.__dict__.get('_get_logprobs_tensors') is not self.installed:
            raise RuntimeError('native logprob adapter ownership changed')
        del self.sampler._get_logprobs_tensors
        self.closed = True
