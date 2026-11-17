"""Use vendor post/comb from FP32 projected mixes; retain SG pre and lag outside."""
import torch
import custom_ops
import datetime,hashlib,json,os
from pathlib import Path
_G112_VERIFIED = False


def split(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6, *, residual, fallback):
    b, s, _ = mixes.shape
    n = b * s
    if mixes.device.type != 'npu' or hc_mult != 4 or sinkhorn_iters != 20 or n < 512:
        return fallback(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
    assert residual.shape == (n, 4, 5120) and residual.dtype == torch.bfloat16
    assert residual.device == mixes.device and residual.is_contiguous()
    flat = mixes.reshape(n, 24).float().contiguous()
    scale = hc_scale.float().contiguous()
    base = hc_base.float().contiguous()
    # SG input mixes already contain the FP32 inverse RMS factor.
    ones = torch.ones((n, 1), device=mixes.device, dtype=torch.float32)
    unused_y, post, comb = torch.ops.custom.npu_hc_pre_sinkhorn(
        flat, ones, scale, base, residual,
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=eps,
    )
    assert post.dtype == comb.dtype == torch.float32
    assert post.numel() == n * 4 and comb.numel() == n * 16
    # The current-pre y is not used by SG's lagged combine. Keep the original
    # pre formula independently, so no current-pre output enters that combine.
    del unused_y
    pre = torch.sigmoid(flat[:, :4] * scale[0] + base[:4]) + eps
    outputs = (
        pre.reshape(b, s, 4).to(mixes.dtype),
        post.reshape(b, s, 4).to(mixes.dtype),
        comb.reshape(b, s, 4, 4).to(mixes.dtype),
    )
    global _G112_VERIFIED
    if not _G112_VERIFIED:
        expected = fallback(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
        errors = [float((a - b).abs().max()) for a, b in zip(outputs, expected)]
        assert all(torch.allclose(a, b, atol=2e-5, rtol=2e-5) for a, b in zip(outputs, expected)), errors
        rank = torch.distributed.get_rank()
        directory = Path('/task/sglang_penalty133/vendor_hits')
        directory.mkdir(exist_ok=True)
        value = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'rank': rank, 'pid': os.getpid(), 'rows': n, 'residual_shape': list(residual.shape), 'dtype': str(residual.dtype), 'atol_rtol': [2e-5, 2e-5], 'actual_input_strict_passed': True, 'max_abs': errors, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'shadow_disabled_after_record': True, 'eps': eps, 'sinkhorn_iters': sinkhorn_iters}
        path = directory / ('rank' + str(rank) + '.json')
        assert not path.exists(), 'Existing vendor-hit evidence; do not overwrite'
        path.write_text(json.dumps(value, indent=2) + '\n')
        print('G112_VENDOR_SINKHORN_ACTUAL_INPUT_PASS ' + json.dumps(value), flush=True)
        _G112_VERIFIED = True
    return outputs
