# SPDX-License-Identifier: Apache-2.0
"""Create the validated empty state for one bounded GLM text Worker request.

This uses the existing KDA and DSA state contracts, not the V4 cache engine.
It does not allocate paged serving caches, enable concurrent requests, or
claim production memory capacity.
"""
import torch

from .cache import state_from_aligned_prefix
from .request_state import MixedRequestState


def initial_worker_text_state(runner, request_id):
    """Build and validate a fresh zero-position state on the runner's device."""
    if not isinstance(request_id,str) or not request_id:
        raise ValueError('nonempty request identity required')
    config=runner.stack.config;layout=runner.stack.layout;layers=runner.stack.layers
    if len(layers)!=len(config.layers):
        raise ValueError('runner layers do not match checkpoint configuration')
    device=runner.boundary.norm_weight.device
    states={}
    for spec,layer in zip(config.layers,layers,strict=True):
        if spec.layer_id!=layer.layer_id:
            raise ValueError('runner layer order differs from checkpoint')
        if spec.attention=='linear_attention':
            states[spec.layer_id]=layer.attention.initial_state()
        elif spec.attention=='deepseek_sparse_attention':
            latent=torch.empty((0,layout.latent_dim),dtype=torch.bfloat16,device=device)
            compressed=torch.empty((0,layout.index_dim),dtype=torch.bfloat16,device=device)
            states[spec.layer_id]=state_from_aligned_prefix(latent,compressed,layout)
        else:
            raise ValueError('unsupported GLM attention type')
    state=MixedRequestState.capture(request_id,config,layout,states)
    if state.tokens_seen!=0:
        raise ValueError('fresh worker state must start at token zero')
    return state
