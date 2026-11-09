"""910C device implementations for the unchanged original sampling interfaces.

All numerical work stays on NPU, using FP32 and integer operations. There is
no CPU/FP64 reference fallback. This is a compatibility implementation, not a
claim of measured performance or graph support. Device validation is required.
Philox4x32-10, RNG namespaces, 31-bit weights and two-level column ordering
follow the pinned original Ascend sampler rather than its FP64 test oracle.
"""
import torch
from deepinfer.npu_ops.sampling_validation910c import assert_nonempty_sampling

from ..common import DRAFT_POS_HASH_MUL, DRAFT_PROB_PENALTY


def _check(*values):
    tensors = [x for x in values if isinstance(x, torch.Tensor)]
    if not tensors or any(x.device.type != 'npu' or x.dtype == torch.float64 for x in tensors):
        raise TypeError('910C sampling requires NPU tensors without FP64')
    if any(x.device != tensors[0].device for x in tensors):
        raise ValueError('910C operator inputs must share a device')


def _philox(seed, position, case):
    # int64 arithmetic emulates the exact uint32 words, including overflow.
    mask = 0xffffffff
    sequence = (position.to(torch.int32) * 3 + case).to(torch.int64)
    key0, key1 = seed & mask, (seed >> 32) & mask
    c0, c1 = torch.zeros_like(seed), torch.zeros_like(seed)
    c2, c3 = sequence & mask, (sequence >> 32) & mask
    for _ in range(10):
        a, b = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c1, c2, c3 = (((b >> 32) & mask) ^ c1 ^ key0,
                           b & mask, ((a >> 32) & mask) ^ c3 ^ key1, a & mask)
        key0, key1 = (key0 + 0x9E3779B9) & mask, (key1 + 0xBB67AE85) & mask
    return c0, c1


def _sample_weights(weights, seeds, positions, case, fallback):
    from deepinfer import envs
    if envs.DEEPINFER_BATCH_INVARIANT:
        from deepinfer.npu_ops.sampling_draw910c import draw
        return draw(weights, seeds, positions, case, fallback)
    q, width = weights.shape
    block = 1 << (width - 1).bit_length()
    slots = 4096 if width >= 8192 else min(block, 2048)
    groups = (width + slots - 1) // slots
    padded = torch.nn.functional.pad(weights, (0, groups * slots - width))
    folded = padded.reshape(q, groups, slots).sum(1)
    cdf = folded.cumsum(1)
    total = cdf[:, -1]
    safe_total = total.clamp_min(1)
    low, high = _philox(seeds, positions, case)
    # Unsigned 64-bit modulo without overflowing signed int64 multiplication.
    target = ((high % safe_total) * ((1 << 32) % safe_total) + low) % safe_total
    offset = (cdf <= target[:, None]).sum(1).clamp_max(slots - 1).long()
    before = cdf.gather(1, (offset - 1).clamp_min(0)[:, None]).squeeze(1)
    target = target - torch.where(offset == 0, 0, before)
    columns = offset[:, None] + torch.arange(groups, device=weights.device)[None] * slots
    local = padded.gather(1, columns).cumsum(1)
    group = (local <= target[:, None]).sum(1)
    selected = offset + group * slots
    return torch.where(total > 0, selected, fallback).to(torch.int32), total


def _weights(probs):
    # Original SIMD conversion saturates a scaled FP32 value to signed int32.
    return (probs * float(1 << 31)).to(torch.int64).clamp(0, (1 << 31) - 1)


def fused_sample_tl(vocab_size, do_sample, do_reject_sampling):
    def kernel(logits, temperatures, top_ps, top_ks, positions, seeds, is_verify,
               logprobs, candidates, draft_lp, read_rows, draft_ids, accepted,
               selected_draft_lp, accept_probs, seed_offset, workspace):
        _check(logits, temperatures, top_ps, top_ks)
        from deepinfer.model_executor.models.decode_trace910c import sampler_before, sampler_after
        observed=sampler_before(logits=logits,temperatures=temperatures,top_ps=top_ps,
            top_ks=top_ks,positions=positions,seeds=seeds,is_verify=is_verify,read_rows=read_rows,
            previous=draft_lp if do_reject_sampling else None,
            previous_tokens=draft_ids if do_reject_sampling else None)
        # Shift before temperature scaling, preserving small differences in
        # high-magnitude logits and the original read-only logits contract.
        shifted = (logits - logits.amax(1, keepdim=True)) / temperatures[:, None]
        probs = torch.softmax(shifted, dim=-1)
        ordered, indices = torch.sort(probs, dim=-1, descending=True, stable=True)
        ranks = torch.arange(vocab_size, device=logits.device)[None]
        removed = ((ordered.cumsum(1) - ordered) > top_ps[:, None])
        removed |= (top_ks[:, None] >= 0) & (ranks >= top_ks[:, None])
        mask = torch.empty_like(removed).scatter_(1, indices, removed)
        logprobs.copy_(torch.log_softmax(shifted.masked_fill(mask, -torch.inf), dim=-1))
        if not do_sample:
            sampler_after(observed,logprobs=logprobs)
            return
        seeds = seeds.view(torch.int64) + seed_offset
        fallback = logprobs.argmax(1)
        from deepinfer import envs
        native_draw = None
        if envs.DEEPINFER_BATCH_INVARIANT:
            from deepinfer.npu_ops.sampling_draw910c import draw_logprobs
            native_draw = draw_logprobs
        if not do_reject_sampling:
            sampled, total = (_sample_weights(_weights(logprobs.exp()), seeds, positions, 0, fallback)
                if native_draw is None else native_draw(logprobs, seeds, positions, 0, fallback))
            assert_nonempty_sampling(total)
            candidates.copy_(sampled)
            sampler_after(observed,logprobs=logprobs,candidates=candidates)
            return
        previous = draft_lp.index_select(0, read_rows.long())
        tokens = draft_ids.index_select(0, read_rows.long())
        safe = tokens.clamp_min(0).long()[:, None]
        old_selected = previous.gather(1, safe).squeeze(1)
        selected_draft_lp.copy_(torch.where(tokens >= 0, old_selected, -torch.inf))
        current_selected = logprobs.gather(1, safe).squeeze(1)
        low, _ = _philox(seeds, positions, 2)
        threshold = ((current_selected - old_selected).clamp_max(0).exp() * float(1 << 33)).to(torch.int64)
        decision = (tokens >= 0) & (((low << 1) | 1) < threshold)
        accepted.copy_(torch.where(is_verify, decision, True))
        if native_draw is None:
            probs = logprobs.exp()
            residual = (probs - (previous - DRAFT_PROB_PENALTY).exp()).clamp_min(0)
            weights = _weights(torch.where(is_verify[:, None], residual, probs))
            sampled, total = _sample_weights(weights, seeds, positions, is_verify.to(torch.int32), fallback)
        else:
            sampled, total = native_draw(logprobs, seeds, positions,
                is_verify.to(torch.int32), fallback, previous=previous, verify=is_verify)
        accept_probs.copy_(torch.where(is_verify, (1 - total.float() / float(1 << 31)).clamp_min(0), 1))
        candidates.copy_(torch.where(is_verify & decision, tokens, sampled))
        sampler_after(observed,logprobs=logprobs,candidates=candidates,accepted=accepted,
                      selected_draft_lp=selected_draft_lp,accept_probs=accept_probs)
    return kernel


def prepare_verification_metadata_tl(num_steps):
    def kernel(ftp, lengths, request, local, verify):
        _check(ftp, lengths)
        rows = torch.arange(request.numel(), dtype=torch.int32, device=ftp.device)
        req = (rows[:, None] >= ftp[None]).sum(1).long() - 1
        request.copy_(req)
        local.copy_(rows - ftp.index_select(0, req))
        verify.copy_(local != lengths.index_select(0, req) - 1)
    return kernel


def expand_sampling_params_tl():
    def kernel(temp, top_p, top_k, positions, seeds, request, local,
               out_temp, out_p, out_k, out_pos, out_seeds):
        _check(temp, request, local)
        req = request.long()
        for source, target in ((temp, out_temp), (top_p, out_p), (top_k, out_k)):
            target.copy_(source.index_select(0, req))
        out_pos.copy_(positions.index_select(0, req) + local * DRAFT_POS_HASH_MUL)
        out_seeds.view(torch.int64).copy_(seeds.view(torch.int64).index_select(0, req))
    return kernel


def map_lookup_tl(width, mode, value_dtype):
    def kernel(key, value, query, out):
        _check(value, query)
        if mode == 'by_key':
            result = torch.where(key == query, value, -torch.inf).amax(1, keepdim=True)
        else:
            valid = (query >= 0) & (query < width)
            result = value.gather(1, query.clamp(0, width - 1).long())
            result = torch.where(valid, result, -torch.inf if value.dtype.is_floating_point else -1)
        out.copy_(result)
    return kernel


def remap_draft_topk_tl(width, vocab_size):
    def kernel(old_ids, old_lp, tokens, new_ids, read_rows, aligned, local_ids, identity):
        _check(old_ids, old_lp, new_ids, read_rows)
        req = read_rows.long()
        dense = old_lp.new_full((new_ids.shape[0], vocab_size), -torch.inf)
        dense.scatter_(1, old_ids.index_select(0, req).long(), old_lp.index_select(0, req))
        aligned.copy_(dense.gather(1, new_ids.long()))
        matches = new_ids == tokens.index_select(0, req)[:, None]
        local_ids.copy_(torch.where(matches.any(1), matches.int().argmax(1), -1))
        identity.copy_(torch.arange(new_ids.shape[0], device=new_ids.device))
    return kernel


def get_accept_result_tl(has_golden):
    def kernel(raw, golden, pool, read_rows, candidate, verify, ftp, request, tokens, accept, final):
        _check(raw, candidate, verify, ftp, request)
        decision = raw
        if has_golden:
            forced = golden != -1
            decision = torch.where(forced, golden == pool.index_select(0, read_rows.long()), decision)
            candidate = torch.where(forced, golden, candidate)
        blocked = ~(verify & decision)
        prefix = torch.nn.functional.pad(blocked.int().cumsum(0), (1, 0))
        alive = prefix[:-1] == prefix.index_select(0, ftp.index_select(0, request.long()).long())
        tokens.copy_(candidate)
        accept.copy_(alive & verify & decision)
        final.copy_(alive & blocked)
    return kernel


def finalize_verification_tl(has_ssp, has_confidence, num_steps, num_stop_tokens):
    def kernel(accept, final, tokens, lp, ssp, top_lp, top_ids, ftp, lengths, request, verify,
               probs, confidence, read_rows, stops, offsets, generated, verified, error, emitted,
               out_tokens, out_lp, out_ssp, out_top_lp, out_top_ids,
               verified_tokens, verified_lp, verified_ssp, verified_top_lp, verified_top_ids):
        _check(accept, final, tokens, lp, ftp)
        rows = torch.arange(tokens.numel(), dtype=torch.int32, device=tokens.device)
        req = request.long()
        final_row = torch.full_like(ftp, -1)
        final_row.scatter_reduce_(0, req, torch.where(final, rows, -1), reduce='amax', include_self=True)
        index = final_row.long()
        offsets.copy_(final_row - ftp)
        generated.zero_(); verified.zero_()
        step = (rows - ftp.index_select(0, req)).clamp_max(num_steps - 1).long()
        generated.scatter_add_(0, step, accept.int())
        verified.scatter_add_(0, step, verify.int())
        out_tokens.copy_(tokens.index_select(0, index)); out_lp.copy_(lp.index_select(0, index))
        verified_tokens.copy_(torch.where(accept, tokens, -1))
        verified_lp.copy_(torch.where(accept, lp, 0))
        if has_ssp:
            out_ssp.copy_(ssp.index_select(0, index))
            verified_ssp.copy_(torch.where(accept, ssp, 0))
        out_top_lp.copy_(top_lp.index_select(0, index)); out_top_ids.copy_(top_ids.index_select(0, index))
        verified_top_lp.copy_(torch.where(accept[:, None], top_lp, 0))
        verified_top_ids.copy_(torch.where(accept[:, None], top_ids, 0))
        error.zero_()
        if has_confidence:
            rates = confidence.index_select(0, read_rows.long())
            error.copy_(torch.where(verify, (probs - rates).abs(), 0).sum().reshape_as(error))
            hits = (tokens[:, None] == stops.index_select(0, req)).any(1) & (accept | final)
            counts = torch.zeros_like(ftp).scatter_add_(0, req, hits.int())
            emitted.copy_(counts > 0)
    return kernel


def gather_tl(width, value_dtype, index_dtype, stride):
    def kernel(source, index, output):
        _check(source, index, output)
        valid = (index >= 0) & (index < source.shape[0])
        value = source.index_select(0, index.clamp(0, source.shape[0] - 1).long())
        output.copy_(torch.where(valid[:, None], value, 0))
    return kernel


def sum_smaller_probs_tl(vocab_size):
    def kernel(logprobs, tokens, output):
        _check(logprobs, tokens, output)
        pivot = logprobs.gather(1, tokens.clamp_min(0).long()[:, None])
        columns = torch.arange(vocab_size, device=logprobs.device)[None]
        selected = (logprobs < pivot) | ((logprobs == pivot) & (columns <= tokens[:, None]))
        output.copy_(torch.where(tokens >= 0, (logprobs.exp() * selected).sum(1), 0))
    return kernel


def store_mtp_info_tl(nextn, width, has_index, has_confidence):
    def kernel(slots, tokens, lp, indices, rates, token_pool, lp_pool, index_pool, rate_pool):
        _check(slots, tokens, lp, token_pool, lp_pool)
        rows = (slots[:, None] * nextn + torch.arange(nextn, device=slots.device)[None]).flatten().long()
        valid = (slots[:, None] != 0).expand(-1, nextn).flatten()
        for source, target in ((tokens, token_pool), (lp, lp_pool),
                               *(([(indices, index_pool)] if has_index else [])),
                               *(([(rates, rate_pool)] if has_confidence else []))):
            payload = source.reshape(rows.numel(), *source.shape[2:])
            before = target.index_select(0, rows)
            keep = valid.reshape((valid.numel(),) + (1,) * (payload.ndim - 1))
            target.index_copy_(0, rows, torch.where(keep, payload, before))
    return kernel


def apply_token_bitmask_by_step_inplace_tl(vocab_size, has_ftp):
    def kernel(logits, bitmask, indices, ftp, step):
        _check(logits, bitmask, indices)
        columns = torch.arange(vocab_size, device=logits.device)
        for number in range(indices.numel()):
            req = indices[number:number + 1].long()
            row = ftp.index_select(0, req) + step if has_ftp else req
            if has_ftp:
                padded = torch.cat((ftp, ftp.new_full((1,), logits.shape[0])))
                end = padded.index_select(0, req + 1)
                valid = (row >= 0) & (row < end)
            else:
                valid = (row >= 0) & (row < logits.shape[0])
            safe_row = row.clamp(0, logits.shape[0] - 1).long()
            words = bitmask.index_select(0, req).index_select(1, columns // 32)
            allowed = ((words >> (columns % 32)) & 1).bool()
            values = logits.index_select(0, safe_row)
            values[:, :vocab_size] = torch.where(allowed | ~valid[:, None], values[:, :vocab_size], -torch.inf)
            logits.index_copy_(0, safe_row, values)
    return kernel


def compute_seqlens_k_zos_tl(nextn):
    def kernel(k, q, remap, previous):
        _check(k, q, remap, previous)
        accepted = previous.index_select(0, remap.clamp_min(0).long())
        k.add_(q - torch.where(remap >= 0, nextn - accepted, 0))
    return kernel
