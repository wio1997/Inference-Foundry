"""Original confidence-budget allocation using NPU FP32/integer operations."""
import torch

from .sampling import _check


def compute_seqlens_q_by_confidence_rates_tl(nextn: int):
    if type(nextn) is not int or nextn < 1:
        raise ValueError('original confidence allocation requires positive nextn')

    def run(remap, prev_confidence_rates, cum_rates, seqlens_q, total_q):
        _check(remap, prev_confidence_rates, cum_rates, seqlens_q)
        count = remap.numel()
        if (remap.shape != (count,) or seqlens_q.shape != (count,)
                or cum_rates.shape != (count, nextn) or prev_confidence_rates.shape[1] != nextn
                or not count <= total_q <= count * (nextn + 1)):
            raise ValueError('confidence allocation differs from original query budget')
        gathered = prev_confidence_rates.index_select(0, remap.clamp_min(0).long())
        acc = (remap != -1).to(torch.float32)
        for position in range(nextn):
            acc = acc * gathered[:, position]
            cum_rates[:, position].copy_(acc)
        # Stable flat-index ordering is the original tie break, including all
        # zero-confidence fresh rows. Allocate only the original extra budget.
        order = torch.argsort(cum_rates.reshape(-1), descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(0, order, torch.arange(order.numel(), dtype=order.dtype, device=order.device))
        wins = (ranks.reshape(count, nextn) < total_q - count).sum(1)
        seqlens_q.copy_((1 + wins).to(seqlens_q.dtype))

    return run
