# SPDX-License-Identifier: Apache-2.0
"""Eligibility for one page-aligned final-chunk YOCO replay candidate."""


def eligible(model, owned, slots, chunks, hits, *, internal, keep=256):
    """Return local eligibility; caller must reach the same decision on all EP ranks."""
    if internal or len(owned)!=4 or slots!=list(range(4)) or len(chunks)!=4 or len(hits)!=4:
        return False
    if (type(keep) is not int or keep not in (140,256) or not model.tail_ffn or
            not model.sequence_parallel or not model.layers[0].context_parallel or
            model.sp.world!=8 or model.dspark is None):
        return False
    if keep < model.cfg['sliding_window']+model.dspark.nextn+1:
        return False
    if any(slot not in model.restored_prefix_slots or hits[slot]<=0 for slot in slots):
        return False
    if any(not req.get('ignore_eos',False) for _,req in owned):
        return False
    ends=[model.next_positions[slot]+len(chunk) for slot,chunk in zip(slots,chunks)]
    if (any(len(chunk)<keep for chunk in chunks) or max(map(len,chunks))<=keep or len(set(ends))!=1 or
            ends[0]%128 or ends[0]-keep<=model.cfg['index_topk'] or
            any(end!=len(owned[slot][1]['token_ids']) for slot,end in zip(slots,ends))):
        return False
    return True
