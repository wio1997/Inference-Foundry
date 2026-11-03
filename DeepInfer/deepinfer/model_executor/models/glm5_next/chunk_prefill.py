# SPDX-License-Identifier: Apache-2.0
"""Opt-in single-sequence GLM KDA chunk prefill; experimental, not a text runner."""
import math
import torch

@torch.inference_mode()
def chunk_prefill(q, k, v, raw_gate, raw_beta, state, a_log, dt_bias, *, lower_bound=-5.0):
    """Run at most one 64-token chunk and return output plus a new FP32 VK state.

    Inputs are projected and causally convolved tensors for one sequence. The
    caller owns projection, cache commit and model integration. No input state
    is mutated, and no precision or performance guarantee is implied.
    """
    if not isinstance(q, torch.Tensor) or q.device.type != 'npu' or q.dtype != torch.bfloat16 or q.ndim != 3:
        raise ValueError('expected NPU BF16 Q [tokens, heads, 128]')
    t, h, dim = q.shape
    if not 1 <= t <= 64 or h < 1 or dim != 128:
        raise ValueError('expected 1..64 tokens and 128-dimensional heads')
    tensors = (k, v, raw_gate, raw_beta, state, a_log, dt_bias)
    if not all(isinstance(x, torch.Tensor) and x.device == q.device for x in tensors):
        raise ValueError('all inputs must be tensors on the same device')
    if any(x.shape != q.shape or x.dtype != q.dtype for x in (k, v, raw_gate)):
        raise ValueError('K/V/gate must match Q geometry and dtype')
    if raw_beta.shape != (t, h) or raw_beta.dtype != q.dtype:
        raise ValueError('beta must be BF16 [tokens, heads]')
    if state.shape != (h, dim, dim) or state.dtype != torch.float32:
        raise ValueError('state must be FP32 [heads, value_dim, key_dim]')
    if a_log.numel() != h or dt_bias.numel() != h * dim:
        raise ValueError('gate parameters do not match head geometry')
    if not math.isfinite(lower_bound) or not -5 <= lower_bound < 0:
        raise ValueError('bounded gate lower_bound must be in [-5,0)')
    from vllm_ascend.ops.kda import run_chunk_kda
    output, final_state = run_chunk_kda(
        q.unsqueeze(0).contiguous(), k.unsqueeze(0).contiguous(),
        v.unsqueeze(0).contiguous(), raw_gate.unsqueeze(0).contiguous(),
        raw_beta.float().sigmoid().unsqueeze(0).contiguous(),
        state.unsqueeze(0).clone().contiguous(), (0, t), (0, 0),
        a_log.reshape(-1).float().contiguous(), dt_bias.reshape(-1).float().contiguous(),
        lower_bound=lower_bound,
    )
    if output.shape != (1, t, h, dim) or final_state.shape != (1, h, dim, dim):
        raise RuntimeError('chunk kernel returned unexpected geometry')
    return output.squeeze(0), final_state.squeeze(0)

@torch.inference_mode()
def chunk_prefill_sharded(q, k, v, raw_gate, raw_beta, state, a_log, dt_bias, *, shards, lower_bound=-5.0):
    """Reconstruct one replicated KDA result from explicit equal head shards.

    Shards are evaluated locally; this is a correctness experiment, not TP or
    a serving prefill implementation. The caller retains ownership of state.
    """
    if type(shards) is not int or shards < 1 or q.ndim != 3 or q.shape[1] % shards:
        raise ValueError('explicit equal head shards required')
    _, heads, dim = q.shape
    if a_log.numel() != heads or dt_bias.numel() != heads * dim:
        raise ValueError('gate parameters do not match head geometry')
    biases = dt_bias.reshape(heads, dim)
    outputs, states = [], []
    for index in range(shards):
        lo, hi = index * (heads // shards), (index + 1) * (heads // shards)
        output, next_state = chunk_prefill(
            q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], raw_gate[:, lo:hi],
            raw_beta[:, lo:hi], state[lo:hi], a_log.reshape(-1)[lo:hi], biases[lo:hi],
            lower_bound=lower_bound,
        )
        outputs.append(output)
        states.append(next_state)
    return torch.cat(outputs, dim=1), torch.cat(states, dim=0)
