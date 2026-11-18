"""Derived fixed-image native router; only CUDA libdevice lowering is adapted to Ascend."""
import triton
import triton.language as tl
@triton.jit
def _ascend_log1p(x):
    u=1.0+x
    exact=u==1.0
    return tl.where(exact,x,x*tl.log(u)/tl.where(exact,1.0,u-1.0))

@triton.jit
def _router_triton_kernel(scores_ptr, bias_ptr, bias_alt_ptr, input_ids_ptr, num_token_non_padded_ptr, out_weights_ptr, out_indices_ptr, out_packed_ptr, M, routed_scaling_factor, moe_softcapping, N: tl.constexpr, K: tl.constexpr, K_ROUTED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, N_GROUP: tl.constexpr, TOPK_GROUP: tl.constexpr, EXPERTS_PER_GROUP: tl.constexpr, BLOCK_G: tl.constexpr, SCORING_FUNC: tl.constexpr, SQRTSOFTPLUS_LOG1P: tl.constexpr, HAS_SOFTCAP: tl.constexpr, RENORMALIZE: tl.constexpr, APPLY_SCALE: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_TOKEN_BIAS: tl.constexpr, BIAS_ALT_TOKEN_ID: tl.constexpr, HAS_PADDING: tl.constexpr, HAS_PACKED: tl.constexpr, RENORMALIZE_EPSILON: tl.constexpr, USE_PDL: tl.constexpr, SCORES_SPLIT_K: tl.constexpr, stride_bias, stride_bias_alt, stride_input_ids, stride_sm, stride_sn, stride_wm, stride_wk, stride_im, stride_ik, stride_pm, stride_pk, stride_split) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n * stride_bias, mask=mask_n, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros([BLOCK_N], dtype=tl.float32)
    if HAS_TOKEN_BIAS:
        bias_alt = tl.load(bias_alt_ptr + offs_n * stride_bias_alt, mask=mask_n, other=0.0).to(tl.float32)
    live_m = mask_m
    if HAS_PADDING:
        live_m = live_m & (offs_m < tl.load(num_token_non_padded_ptr))
    row_bias = bias[None, :]
    if HAS_TOKEN_BIAS:
        input_ids = tl.load(input_ids_ptr + offs_m * stride_input_ids, mask=live_m, other=0)
        row_bias = tl.where((input_ids == BIAS_ALT_TOKEN_ID)[:, None], bias_alt[None, :], row_bias)
    row_ptr = scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    mask2d = live_m[:, None] & mask_n[None, :]
    if SCORES_SPLIT_K == 1:
        scores = tl.load(row_ptr, mask=mask2d, other=0.0).to(tl.float32)
    else:
        scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for split in tl.static_range(SCORES_SPLIT_K):
            scores += tl.load(row_ptr + split * stride_split, mask=mask2d, other=0.0).to(tl.float32)
    if SCORING_FUNC == 0:
        activated = tl.sigmoid(scores)
        biased = activated + row_bias
    elif SCORING_FUNC == 1:
        if SQRTSOFTPLUS_LOG1P:
            sp = tl.where(scores > 20.0, scores, _ascend_log1p(tl.exp(scores)))
            activated = tl.sqrt(sp)
        else:
            z = tl.exp(-tl.abs(scores))
            u = 1.0 + z
            exact = u == 1.0
            log1p_z = tl.where(exact, z, z * tl.log(u) / tl.where(exact, 1.0, u - 1.0))
            sp = tl.maximum(scores, 0.0) + log1p_z
            activated = tl.sqrt(sp)
        biased = activated + row_bias
    else:
        logit = scores
        if HAS_SOFTCAP:
            z = logit / moe_softcapping
            logit = moe_softcapping * (2.0 * tl.sigmoid(2.0 * z) - 1.0)
        biased = logit + row_bias
        biased = tl.where(mask_n[None, :], biased, -float('inf'))
        row_max = tl.max(biased, axis=1)[:, None]
        exp_row = tl.where(mask_n[None, :], tl.exp(biased - row_max), 0.0)
        row_sum = tl.sum(exp_row, axis=1)[:, None]
        activated = exp_row / row_sum
    biased = tl.where(mask_n[None, :], biased, -float('inf'))
    if SCORING_FUNC == 1 and SQRTSOFTPLUS_LOG1P:
        biased = tl.where(biased == biased, biased, float('inf'))
    else:
        biased = tl.where(biased == biased, biased, -1e+30)
    if N_GROUP > 1:
        offs_g = tl.arange(0, BLOCK_G)
        group_of_n = offs_n // EXPERTS_PER_GROUP
        group_score = tl.full([BLOCK_M, BLOCK_G], -float('inf'), dtype=tl.float32)
        for g in tl.static_range(N_GROUP):
            in_g = (group_of_n[None, :] == g) & mask_n[None, :]
            vals = tl.where(in_g, biased, -float('inf'))
            top1 = tl.max(vals, axis=1)[:, None]
            vals2 = tl.where(vals >= top1, -float('inf'), vals)
            top2 = tl.max(vals2, axis=1)[:, None]
            group_score = tl.where(offs_g[None, :] == g, top1 + top2, group_score)
        gcur = group_score
        keep = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for _i in tl.static_range(TOPK_GROUP):
            gmax = tl.max(gcur, axis=1)[:, None]
            glane = tl.where(gcur == gmax, offs_g[None, :], N_GROUP + 1)
            win_g = tl.min(glane, axis=1)[:, None]
            keep = tl.where(group_of_n[None, :] == win_g, 1.0, keep)
            gcur = tl.where(offs_g[None, :] == win_g, -float('inf'), gcur)
        biased = tl.where(keep > 0.0, biased, -float('inf'))
    offs_k = tl.arange(0, BLOCK_K)
    mask_k_total = offs_k < K
    mask_k_routed = offs_k < K_ROUTED
    selected_vals = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)
    cur = biased
    remaining = tl.broadcast_to(mask_n[None, :], (BLOCK_M, BLOCK_N))
    for k in tl.static_range(K_ROUTED):
        max_val = tl.max(cur, axis=1)[:, None]
        is_max = remaining & (cur == max_val)
        lane_id = tl.where(is_max, offs_n[None, :], N + 1)
        win_lane = tl.min(lane_id, axis=1)[:, None].to(tl.int32)
        win_activated = tl.sum(tl.where(offs_n[None, :] == win_lane, activated, 0.0), axis=1)[:, None]
        slot = offs_k[None, :] == k
        selected_vals = tl.where(slot, win_activated, selected_vals)
        selected_idx = tl.where(slot, win_lane, selected_idx)
        remaining = remaining & (offs_n[None, :] != win_lane)
        cur = tl.where(remaining, cur, -float('inf'))
    routed_sum = tl.sum(tl.where(mask_k_routed[None, :], selected_vals, 0.0), axis=1)[:, None]
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    if RENORMALIZE:
        if RENORMALIZE_EPSILON > 0.0:
            norm = routed_sum + RENORMALIZE_EPSILON
        else:
            norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)
        selected_vals = selected_vals / norm
    if APPLY_SCALE:
        selected_vals = selected_vals * routed_scaling_factor
    if K_ROUTED < K:
        is_shared = (offs_k[None, :] >= K_ROUTED) & mask_k_total[None, :]
        shared_weight = 1.0 if APPLY_SCALE else 1.0 / routed_scaling_factor
        shared_idx = (N + (offs_k - K_ROUTED)).to(tl.int32)[None, :]
        selected_vals = tl.where(is_shared, shared_weight, selected_vals)
        selected_idx = tl.where(is_shared, shared_idx, selected_idx)
    if HAS_PADDING:
        selected_vals = tl.where(live_m[:, None], selected_vals, 0.0)
        selected_idx = tl.where(live_m[:, None], selected_idx, -1)
    out_w_ptr = out_weights_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk
    out_i_ptr = out_indices_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    store_mask = mask_m[:, None] & mask_k_total[None, :]
    tl.store(out_w_ptr, selected_vals, mask=store_mask)
    tl.store(out_i_ptr, selected_idx, mask=store_mask)
    if HAS_PACKED:
        w_bits = selected_vals.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
        packed = selected_idx << 16 | w_bits & 65535
        out_p_ptr = out_packed_ptr + offs_m[:, None] * stride_pm + offs_k[None, :] * stride_pk
        tl.store(out_p_ptr, packed, mask=store_mask)
