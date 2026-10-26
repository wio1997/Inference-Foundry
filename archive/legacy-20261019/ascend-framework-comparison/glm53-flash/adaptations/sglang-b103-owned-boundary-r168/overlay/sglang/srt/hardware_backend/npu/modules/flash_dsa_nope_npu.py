# SPDX-License-Identifier: Apache-2.0
"""Candidate eager Flash NoPE projection path for B103; not NPU-qualified.

Projection order follows B103's DeepSeek MLA NPU implementation (Apache-2.0),
with Flash's absent rotary tensors handled explicitly. This retains the loaded
W8A8 projection modules. KPool/index cache and numerical gates are prerequisites.
"""
import torch_npu


def prepare_flash_dsa_nope(m, positions, hidden_states, forward_batch,
                          zero_allocator, layer_scatter_modes,
                          prev_topk_indices=None):
    from sglang.srt.runtime_context import get_parallel

    if not getattr(m, '_flash_npu_kpool', False):
        raise ValueError('Flash-only NoPE path requires its architecture marker')
    if m.qk_rope_head_dim != 0 or m.rotary_emb is not None:
        raise ValueError('Flash NoPE route cannot discard rotary features')
    if m.is_nextn or forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend_v2():
        raise ValueError('MTP requires separate state/rollback qualification')
    if get_parallel().dcp_enabled or forward_batch.attn_cp_metadata is not None:
        raise ValueError('Initial Flash NoPE candidate requires CP1/DCP1')
    if m.q_lora_rank is None or m.indexer is None or m.skip_topk:
        raise ValueError('Initial Flash candidate requires a fresh KPool indexer per DSA layer')
    if getattr(m.indexer, 'index_kpool', None) != 4:
        raise ValueError('Expected actual Flash KPool indexer')

    fused = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
    q_raw, k_raw = fused.split([m.q_lora_rank, m.kv_lora_rank], dim=-1)
    q_lora = m.q_a_layernorm(q_raw)
    k_nope = m.kv_a_layernorm(k_raw.unsqueeze(1))
    query = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_nope_head_dim)
    q_nope_out = torch_npu.npu_transpose_batchmatmul(
        query, m.w_kc, perm_x1=(1,0,2), perm_x2=(0,1,2), perm_y=(1,0,2))
    # forward_npu must implement this actual seven-argument NPU call contract.
    topk_indices = m.indexer(hidden_states, q_lora, positions, forward_batch,
                            m.layer_id, layer_scatter_modes, None)
    return (None, None, q_nope_out, k_nope, topk_indices, forward_batch,
            zero_allocator, positions, False)
