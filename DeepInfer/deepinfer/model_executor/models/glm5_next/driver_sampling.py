# SPDX-License-Identifier: Apache-2.0
"""Bounded driver-only sampling through DeepInfer's existing CP broadcast.

The caller supplies the installed sampler and its native metadata, and owns
scheduler output processing. This helper returns a single token only; it does
not provide logprobs, speculative decoding, rollback, or process-loss recovery.
"""
import torch


@torch.inference_mode()
def sample_driver_token(logits, sampling_metadata, sampler):
    """Sample on CP rank zero, broadcast success or failure to all CP ranks.

    All CP ranks must call in the same order. Only rank zero supplies logits,
    metadata and the sampler; all other ranks pass None for those arguments.
    A failed sampling call does not roll back a previously committed model step.
    """
    from deepinfer.model_executor.parallel_utils.parallel_state import PG
    from deepinfer.model_executor.parallel_utils.communication_op import broadcast_tensor_dict

    cp=PG.context_parallel_group();payload=None;failure=None
    if cp.rank()==0:
        try:
            if not isinstance(logits,torch.Tensor) or logits.ndim!=2 or logits.shape[0]!=1 or logits.shape[1]<1:
                raise ValueError('one nonempty logits row required')
            if sampling_metadata is None or sampler is None:
                raise ValueError('native sampling metadata and sampler required')
            if sampling_metadata.max_num_logprobs is not None or sampling_metadata.logprob_token_ids:
                raise ValueError('logprob results are not supported by this boundary')
            output=sampler(logits.clone(),sampling_metadata)
            token_ids=output.sampled_token_ids
            if output.logprobs_tensors is not None or token_ids.shape!=(1,1):
                raise ValueError('sampler returned unsupported output')
            token=int(token_ids.item())
            if not 0<=token<logits.shape[1]:raise ValueError('sampled token outside vocabulary')
            payload={'ok':True,'token':token}
        except Exception as exc:
            failure=exc;payload={'ok':False}
    payload=broadcast_tensor_dict(payload,src=0)
    if not payload['ok']:raise RuntimeError('driver sampling failed') from failure
    return payload['token']


def validate_greedy_metadata(sampling_metadata, sampler):
    """Reject unsupported sampling semantics before a caller commits model state."""
    if sampling_metadata is None or sampler is None:
        raise ValueError('native sampling metadata and sampler required')
    m=sampling_metadata
    if not m.all_greedy or m.all_random or not m.no_penalties or m.max_num_logprobs!=-1 or m.logprob_token_ids:
        raise ValueError('only unpenalized greedy full-logprob sampling is supported')
    if m.allowed_token_ids_mask is not None or m.bad_words_token_ids or m.logitsprocs.non_argmax_invariant or m.logitsprocs.argmax_invariant:
        raise ValueError('logits filters and processors are not supported')


@torch.inference_mode()
def sample_driver_output(logits, sampling_metadata, sampler):
    """Return a single greedy token and its real processed logprob on every CP rank.

    This bounded Worker-output seam requires unfiltered logits, no penalties,
    no logit processors, and the installed sampler's full processed logprobs.
    It does not implement top-logprobs, structured output, prompt logprobs,
    speculation, or non-greedy sampling. Caller owns scheduler post-processing.
    """
    import math
    from deepinfer.model_executor.parallel_utils.parallel_state import PG
    from deepinfer.model_executor.parallel_utils.communication_op import broadcast_tensor_dict
    from deepinfer.sequence import SampledInfo, SamplerOutput

    cp=PG.context_parallel_group();payload=None;failure=None
    if cp.rank()==0:
        try:
            if not isinstance(logits,torch.Tensor) or logits.ndim!=2 or logits.shape[0]!=1 or logits.shape[1]<1:
                raise ValueError('one nonempty logits row required')
            validate_greedy_metadata(sampling_metadata, sampler)
            token_ids,logprobs=sampler.sample(logits.clone(),sampling_metadata,logprobs_mode_override='processed_logprobs')
            if token_ids.numel()!=1 or logprobs is None or logprobs.shape!=logits.shape:
                raise ValueError('sampler returned unsupported output')
            token=int(token_ids.reshape(-1)[0].item())
            if not 0<=token<logits.shape[1]:raise ValueError('sampled token outside vocabulary')
            logprob=float(logprobs[0,token].item())
            if not math.isfinite(logprob):raise ValueError('nonfinite sampled token logprob')
            payload={'ok':True,'token':token,'logprob':logprob}
        except Exception as exc:
            failure=exc;payload={'ok':False}
    payload=broadcast_tensor_dict(payload,src=0)
    if not payload['ok']:raise RuntimeError('driver output sampling failed') from failure
    return SamplerOutput(sampled_token=SampledInfo(bs=1,token_ids=torch.tensor([payload['token']],dtype=torch.int32),
                         token_logprobs=torch.tensor([payload['logprob']],dtype=torch.float32)),post_funcs=[])
