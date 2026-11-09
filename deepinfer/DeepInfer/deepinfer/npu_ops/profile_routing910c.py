"""Original force-random physical expert routing for decode cost profiling."""
import torch


def make_random_routes(rows, topk, *, device, ep_rank, num_experts):
    from deepinfer.model_executor.layers.tilelang_ops.asc910c.sampling import _philox

    if rows < 1 or not 1 <= topk <= min(32, num_experts):
        raise ValueError('invalid original random-gate profile dimensions')
    # The original Ascend kernel initializes lane i with sequence=row*32+i.
    sequence = torch.arange(rows, dtype=torch.int64, device=device)[:, None] * 32
    lane = torch.arange(topk, dtype=torch.int64, device=device)
    sequence = sequence + lane
    first, second = _philox(torch.full_like(sequence, ep_rank), sequence // 3, sequence % 3)
    draws = second.remainder(num_experts - lane)
    # Reverse insertion is equivalent to the original value/lane minimum loop:
    # only later lanes are shifted; already finalized earlier lanes stay fixed.
    for index in range(topk - 2, -1, -1):
        draws[:, index + 1:] += (draws[:, index + 1:] >= draws[:, index:index + 1]).to(draws.dtype)
    random_weights = first.to(torch.float32) * (2.0 ** -32) + 2.0 ** -33
    return draws.to(torch.int32), random_weights


def force_random_routes(ids, weights, flags, *, random_routes):
    if flags.shape != ids.shape[:1] or any(value.shape != ids.shape for value in random_routes):
        raise ValueError('original profile routing buffers differ from local token rows')
    draws, random_weights = random_routes
    return (torch.where(flags[:, None], draws.to(ids.dtype), ids),
            torch.where(flags[:, None], random_weights.to(weights.dtype), weights))
