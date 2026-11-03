# SPDX-License-Identifier: Apache-2.0
"""Opt-in output-sharded gate/up for a checkpoint-backed dense decoder.

The downstream projection and all other decoder components remain replicated.
All ranks must enter the decoder in the same order with identical inputs/state.
This diagnostic adapter is not complete tensor or expert parallelism.
"""
from .decoder import GlmDenseMLP
from .rank_linear import RankW8A8OutputLinear

def install_rank_dense(layer, reader, device, *, rank, world_size, max_intermediate_rows, group=None, max_input_tokens=8):
    if type(layer.layer_id) is not int or layer.layer_id < 0 or type(layer.mlp) is not GlmDenseMLP:
        raise ValueError('checkpoint dense decoder required')
    width=layer.mlp.intermediate_size
    if type(max_intermediate_rows) is not int or max_intermediate_rows < width:
        raise ValueError('explicit intermediate bound required')
    if isinstance(layer.mlp.gate_proj, RankW8A8OutputLinear) or isinstance(layer.mlp.up_proj, RankW8A8OutputLinear):
        raise ValueError('rank projections already installed')
    prefix=f'model.language_model.layers.{layer.layer_id}.mlp'
    options=dict(rank=rank,world_size=world_size,row_start=0,row_stop=width,max_output_rows=max_intermediate_rows,group=group,max_input_tokens=max_input_tokens)
    # Build both first: a loader failure must leave the decoder unchanged.
    gate=RankW8A8OutputLinear(reader,prefix+'.gate_proj',device,**options)
    up=RankW8A8OutputLinear(reader,prefix+'.up_proj',device,**options)
    layer.mlp.gate_proj,layer.mlp.up_proj=gate,up
    return layer
