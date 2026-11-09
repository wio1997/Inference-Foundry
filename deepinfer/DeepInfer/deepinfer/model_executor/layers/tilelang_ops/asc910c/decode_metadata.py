"""910C device factories for the original V4 ZOS input preparation API.

The Python/C++ producer, compressed-token policy and metadata consumers stay
in their original modules. These integer kernels implement the contracts of
asc/{gather_input_ids_zos,gather_engram_tokens_zos,prepare_v4_decode_metadata}.
No sampled token or sequence length is copied back to the host here.
"""
import torch
import triton
import triton.language as tl


def _check(*values):
    if not values or any(value.device.type != 'npu' for value in values):
        raise TypeError('V4 device metadata requires NPU tensors')
    if any(value.device != values[0].device for value in values):
        raise ValueError('V4 device metadata must belong to one NPU')
    if any(value.dtype not in (torch.int32, torch.bool) for value in values):
        raise TypeError('V4 device metadata requires int32/bool buffers')


@triton.jit
def _gather_ids(fresh, sampled, predict, remap, sq, cu, output,
                F0: tl.constexpr, F1: tl.constexpr, P0: tl.constexpr,
                P1: tl.constexpr, MAX_Q: tl.constexpr,
                HAS_PREVIOUS: tl.constexpr, B: tl.constexpr):
    seq = tl.program_id(0)
    j = tl.arange(0, B)
    length = tl.load(sq + seq)
    base = tl.where(seq > 0, tl.load(cu + tl.maximum(seq - 1, 0)), 0)
    previous = tl.load(remap + seq)
    valid = (j < length) & (j < MAX_Q)
    if HAS_PREVIOUS:
        safe_previous = tl.maximum(previous, 0)
        first = tl.load(sampled + safe_previous, previous >= 0, other=0)
        if MAX_Q > 1:
            draft = tl.load(predict + safe_previous * P0 + tl.maximum(j - 1, 0) * P1,
                            (previous >= 0) & (j > 0) & valid, other=0)
        else:
            draft = tl.full((B,), 0, tl.int32)
        new = tl.load(fresh + seq * F0 + j * F1, (previous < 0) & valid, other=0)
        value = tl.where(previous < 0, new, tl.where(j == 0, first, draft))
    else:
        # Empty previous tensors have null device pointers; remove every
        # previous-token access at compile time, including nextn=0 drafts.
        value = tl.load(fresh + seq * F0 + j * F1, valid, other=0)
    tl.store(output + base + j, value, valid)


def gather_input_ids_zos_tl(max_seqlen_q):
    if type(max_seqlen_q) is not int or max_seqlen_q < 1:
        raise ValueError('ZOS query capacity must be positive')

    def run(fresh, sampled, predict, remap, sq, cu, output):
        _check(fresh, sampled, predict, remap, sq, cu, output)
        if sq.numel():
            _gather_ids[(sq.numel(),)](fresh, sampled, predict, remap, sq, cu, output,
                fresh.stride(0), fresh.stride(1), predict.stride(0), predict.stride(1),
                max_seqlen_q, sampled.numel() > 0,
                triton.next_power_of_2(max_seqlen_q), num_warps=4)
    return run


@triton.jit
def _gather_ngrams(win, fresh, sampled, predict, accepted, remap, lookup, sq, cu, output,
                   W0: tl.constexpr, W1: tl.constexpr, F0: tl.constexpr,
                   F1: tl.constexpr, P0: tl.constexpr, P1: tl.constexpr,
                   MAX_Q: tl.constexpr, NGRAM: tl.constexpr, PAD: tl.constexpr,
                   IMAGE: tl.constexpr, HAS_PREVIOUS: tl.constexpr, B: tl.constexpr):
    seq = tl.program_id(0)
    j = tl.program_id(1)
    r = tl.arange(0, B)
    length = tl.load(sq + seq)
    base = tl.where(seq > 0, tl.load(cu + tl.maximum(seq - 1, 0)), 0)
    previous = tl.load(remap + seq)
    old = previous >= 0
    window = MAX_Q + NGRAM - 2
    valid = (j < length) & (r < NGRAM)
    c = j - r
    use_current = c >= 0
    if HAS_PREVIOUS:
        safe_previous = tl.maximum(previous, 0)
        nacc = tl.load(accepted + safe_previous, old, other=0)
        raw_first = tl.load(sampled + safe_previous, old, other=0)
        if MAX_Q > 1:
            raw_draft = tl.load(predict + safe_previous * P0 + tl.maximum(c - 1, 0) * P1,
                                valid & old & (c > 0), other=0)
        else:
            raw_draft = tl.full((B,), 0, tl.int32)
        raw = tl.where(c == 0, raw_first, raw_draft)
        mapped = tl.load(lookup + raw, valid & old & use_current, other=PAD)
    else:
        nacc = tl.full((), 0, tl.int32)
        mapped = tl.full((B,), PAD, tl.int32)
    rejected = (MAX_Q - 1) - nacc
    new = tl.load(fresh + seq * F0 + tl.maximum(c, 0) * F1,
                   valid & ~old & use_current, other=PAD)
    if MAX_Q + NGRAM > 2:
        history = tl.load(win + seq * W0 + tl.maximum(window - rejected + c, 0) * W1,
                           valid & ~use_current, other=PAD)
    else:
        history = tl.full((B,), PAD, tl.int32)
    token = tl.where(use_current, tl.where(old, mapped, new), history)
    # The original image sentinel masks this and all older lookbacks. A
    # prefix sum expresses the same rule without dependent memory masks.
    hit = tl.cumsum(((token == IMAGE) & (r < NGRAM)).to(tl.int32), 0) > 0
    tl.store(output + (base + j) * NGRAM + r, tl.where(hit, PAD, token), valid)


def gather_engram_tokens_zos_tl(max_seqlen_q, max_ngram_size,
                                engram_compressed_pad_token, image_sentinel):
    if any(type(n) is not int or n < 1 for n in (max_seqlen_q, max_ngram_size)):
        raise ValueError('ZOS query and ngram capacities must be positive')

    def run(win, fresh, sampled, predict, accepted, remap, lookup, sq, cu, output):
        _check(win, fresh, sampled, predict, accepted, remap, lookup, sq, cu, output)
        if sq.numel():
            _gather_ngrams[(sq.numel(), max_seqlen_q)](win, fresh, sampled, predict, accepted,
                remap, lookup, sq, cu, output, win.stride(0), win.stride(1),
                fresh.stride(0), fresh.stride(1), predict.stride(0), predict.stride(1),
                max_seqlen_q, max_ngram_size, engram_compressed_pad_token,
                image_sentinel, sampled.numel() > 0,
                triton.next_power_of_2(max_ngram_size), num_warps=4)
    return run


@triton.jit
def _metadata(sq, sk, cu, tables, offsets, states, placeholders, positions,
              valid_tokens, random_gate, fixed_gate, contexts, klen, ca_window,
              swa_window, swa_slots, queries, state_out, compressed_pos, ca_slots,
              MAX_Q: tl.constexpr, MAX_BLOCKS: tl.constexpr, WINDOW: tl.constexpr,
              SWA_BLOCK: tl.constexpr, BLOCK: tl.constexpr, FORCE: tl.constexpr,
              RATIOS: tl.constexpr, DENSE: tl.constexpr, SPARSE: tl.constexpr,
              KSTRIDE: tl.constexpr, CSTRIDE: tl.constexpr,
              POS_STRIDE: tl.constexpr, SLOT_STRIDE: tl.constexpr, B: tl.constexpr):
    seq = tl.program_id(0)
    j = tl.arange(0, B)
    length = tl.load(sq + seq)
    end = tl.load(sk + seq)
    base = tl.where(seq > 0, tl.load(cu + tl.maximum(seq - 1, 0)), 0)
    state = tl.load(states + seq)
    placeholder = tl.load(placeholders + seq) != 0
    bt_offset = tl.load(offsets + seq)
    row = base + j
    pos = end - length + j
    valid = (j < length) & (j < MAX_Q)
    tl.store(positions + row, pos, valid)
    tl.store(valid_tokens + row, ~placeholder, valid)
    tl.store(random_gate + row, FORCE, valid)
    tl.store(fixed_gate + row, False, valid)
    tl.store(contexts + row, pos + 1, valid)
    for i in tl.static_range(len(SPARSE)):
        tl.store(klen + i * KSTRIDE + row, (pos + 1) // SPARSE[i], valid)
    for i in tl.static_range(len(DENSE)):
        tl.store(ca_window + i * CSTRIDE + row * 2,
                 row * MAX_BLOCKS * (BLOCK // DENSE[i]), valid)
        tl.store(ca_window + i * CSTRIDE + row * 2 + 1, (pos + 1) // DENSE[i], valid)
    window_length = tl.minimum(pos + 1, WINDOW)
    tl.store(swa_window + row * 2,
             state * SWA_BLOCK + (pos - window_length + 1) % SWA_BLOCK, valid)
    tl.store(swa_window + row * 2 + 1, window_length, valid)
    tl.store(swa_slots + row, state * SWA_BLOCK + pos % SWA_BLOCK, valid)
    tl.store(queries + row, tl.where(placeholder, 998244353, seq), valid)
    tl.store(state_out + row, state, valid)
    for i in tl.static_range(len(RATIOS)):
        compressed_block = BLOCK // RATIOS[i]
        index = pos // RATIOS[i]
        complete = (pos + 1) % RATIOS[i] == 0
        physical = tl.load(tables + bt_offset + index // compressed_block,
                           valid & complete, other=0)
        tl.store(compressed_pos + i * POS_STRIDE + row,
                 tl.where(complete, index * RATIOS[i], 0), valid)
        tl.store(ca_slots + i * SLOT_STRIDE + row,
                 tl.where(complete, physical * compressed_block + index % compressed_block, -1), valid)


def prepare_v4_decode_metadata_tl(max_seqlen_q, max_num_blocks, window_size,
                                 swa_block_size, block_size, force_load_balance,
                                 ca_ratios, dense_ratios, sparse_ratios):
    if any(type(n) is not int or n < 1 for n in
           (max_seqlen_q, max_num_blocks, window_size, swa_block_size, block_size)):
        raise ValueError('V4 metadata dimensions must be positive')
    ratios, dense, sparse = map(tuple, (ca_ratios, dense_ratios, sparse_ratios))
    if (set(dense) | set(sparse) != set(ratios)
            or any(type(r) is not int or r < 1 or block_size % r for r in ratios)):
        raise ValueError('V4 compression ratios must partition valid block divisors')

    def run(sq, sk, cu, tables, offsets, states, placeholders, positions,
            valid_tokens, random_gate, fixed_gate, contexts, klen, ca_window,
            swa_window, swa_slots, queries, state_out, compressed_pos, ca_slots):
        _check(sq, sk, cu, tables, offsets, states, placeholders, positions,
               valid_tokens, random_gate, fixed_gate, contexts, klen, ca_window,
               swa_window, swa_slots, queries, state_out, compressed_pos, ca_slots)
        if sq.numel():
            _metadata[(sq.numel(),)](sq, sk, cu, tables, offsets, states, placeholders,
                positions, valid_tokens, random_gate, fixed_gate, contexts, klen,
                ca_window, swa_window, swa_slots, queries, state_out, compressed_pos,
                ca_slots, max_seqlen_q, max_num_blocks, window_size, swa_block_size,
                block_size, force_load_balance, ratios, dense, sparse,
                klen.stride(0), ca_window.stride(0), compressed_pos.stride(0),
                ca_slots.stride(0), triton.next_power_of_2(max_seqlen_q), num_warps=4)
    return run


@triton.jit
def _copy_tables(sq, cu, tables, offsets, output, ROW_STRIDE: tl.constexpr,
                 COL_STRIDE: tl.constexpr, MAX_BLOCKS: tl.constexpr, B: tl.constexpr):
    seq = tl.program_id(0)
    j = tl.program_id(1)
    col = tl.arange(0, B)
    length = tl.load(sq + seq)
    base = tl.where(seq > 0, tl.load(cu + tl.maximum(seq - 1, 0)), 0)
    first = tl.load(offsets + seq)
    last = tl.load(offsets + seq + 1)
    valid = (j < length) & (col < last - first) & (col < MAX_BLOCKS)
    value = tl.load(tables + first + col, valid, other=0)
    # Preserve unused table columns, as the original Ascend factory does.
    tl.store(output + (base + j) * ROW_STRIDE + col * COL_STRIDE, value, valid)


def copy_block_tables_tl(max_seqlen_q, max_num_blocks):
    if any(type(n) is not int or n < 1 for n in (max_seqlen_q, max_num_blocks)):
        raise ValueError('V4 query/block capacities must be positive')

    def run(sq, cu, tables, offsets, output):
        _check(sq, cu, tables, offsets, output)
        if sq.numel():
            _copy_tables[(sq.numel(), max_seqlen_q)](sq, cu, tables, offsets, output,
                output.stride(0), output.stride(1), max_num_blocks,
                triton.next_power_of_2(max_num_blocks), num_warps=4)
    return run
