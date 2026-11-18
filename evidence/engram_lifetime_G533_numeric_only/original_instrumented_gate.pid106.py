import torch

def rotated_gate(x, kv, q_weight, k_weight, eps, clamp_value, rotation_block, input_ids=None, image_token_id=None):
    """Original-basis gate; the saved value projection is already rotated."""
    hc, dim = x.shape[-2:]
    key, value = kv.split([hc * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc, dim))
    original = (x.float().unflatten(-1, (-1, 32)) @ rotation_block.T).flatten(-2)
    rstd = torch.rsqrt(original.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (original * (q_weight.float() * k_weight.float()) * key).sum(-1) * rstd * dim ** (-0.5)
    magnitude = dot.abs().clamp_min(clamp_value).sqrt()
    negative = dot.view(torch.int32) < 0
    gate = torch.sigmoid(torch.where(negative, -magnitude, magnitude))
    observe('original')
    out = (x.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
    if input_ids is not None:
        out = torch.where((input_ids == image_token_id)[:, None, None], x, out)
    return out
