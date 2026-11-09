"""Real W8A8 full-layer CPU golden, with a separately selected NPU probe.

`--mode cpu` is CPU-only and verifies production layer wiring by substituting
integer CPU matmuls. This is not proof of Ascend INT8 operators or HCCL.
`--mode npu` is prepared for future explicit device authorization only.
"""
import argparse
import ast
import importlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint, FloatLinear
from ascend910c_w8a8_probe import cpu_linear, cpu_expert


def linear(x, weight):
    return F.linear(x.float(), weight.float()).bfloat16()


def norm(x, weight, eps):
    value = x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+eps)*weight.float()
    return value.bfloat16(), value


class GoldenLayer:
    """Independent official mHC reference + integer INT8 CPU computation."""
    def __init__(self, reader, mhc_reference_source):
        self.reader = reader
        self.c = reader.config['text_config']
        source = Path(mhc_reference_source)
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.ref = {'torch': torch}
        exec(compile(tree, str(source), 'exec'), self.ref)

    def op(self, name, *args, **kwargs):
        if name == 'mhc_post':
            # The unchanged official reference names both batch and sequence
            # dimensions in its einsum. This probe uses [sequence, ...], so
            # introduce one batch dimension at the test boundary only.
            return self.ref[name+'_ref'](*(value.unsqueeze(0) for value in args), **kwargs).squeeze(0)
        return self.ref[name+'_ref'](*args, **kwargs)

    def tensor(self, name):
        return self.reader.tensor('layers.0.'+name)[0]

    def project(self, x, name):
        weight, scale = self.reader.linear_payload('layers.0.'+name)
        return linear(x, weight) if scale is None else cpu_linear(x, weight, scale)

    def pre(self, trunk, previous, kind):
        c = self.c
        trunk = trunk.unsqueeze(0)
        value = self.op('mhc_pre_apply_mix', trunk, previous.unsqueeze(0)).squeeze(0)
        mixes = self.op('mhc_pre_norm_fn', trunk, self.tensor(f'hc_{kind}_fn'), None, c['rms_norm_eps'])
        pre, post, comb = self.op('mhc_pre_split_mixes', mixes, self.tensor(f'hc_{kind}_scale'),
            self.tensor(f'hc_{kind}_base'), c['hc_mult'], 2., c['hc_eps'])
        comb = self.op('sinkhorn_normalize', comb, repeat=c['hc_sinkhorn_iters'], eps=c['hc_eps'])
        return value, pre.squeeze(0), post.squeeze(0), comb.squeeze(0)

    def expert(self, x, name):
        payload = [self.reader.linear_payload(f'layers.0.ffn.{name}.w{i}') for i in (1, 3, 2)]
        return cpu_expert(x, payload, self.c['swiglu_limit'])

    def forward(self, trunk, previous):
        c = self.c
        value, pre, post, comb = self.pre(trunk, previous, 'attn')
        x = norm(value, self.tensor('attn_norm.weight'), c['rms_norm_eps'])[0]
        q = self.project(norm(self.project(x, 'attn.wq_a'), self.tensor('attn.q_norm.weight'), c['rms_norm_eps'])[0], 'attn.wq_b')
        q = q.reshape(1, c['num_attention_heads'], c['head_dim'])
        kv = norm(self.project(x, 'attn.wkv'), self.tensor('attn.kv_norm.weight'), c['rms_norm_eps'])[0]
        # Position-zero SWA has one key. Independent closed form includes sink.
        scores = (q.float()*kv.float()[:, None]).sum(-1)*c['head_dim']**-.5
        context = (kv.float()[:, None]*torch.sigmoid(scores-self.tensor('attn.attn_sink').float())[..., None]).bfloat16()
        weight = self.reader.linear_payload('layers.0.attn.wo_a')[0].reshape(c['o_groups'], c['o_lora_rank'], -1)
        projected = torch.einsum('bgr,gdr->bgd', context.reshape(1, c['o_groups'], -1).float(), weight.float()).bfloat16()
        attn = self.project(projected.flatten(1), 'attn.wo_b')
        after = self.op('mhc_post', attn, trunk, post, comb)
        value, next_pre, post, comb = self.pre(after, pre, 'ffn')
        x, x_fp32 = norm(value, self.tensor('ffn_norm.weight'), c['rms_norm_eps'])
        logits = F.linear(x_fp32, self.tensor('ffn.gate.weight').float())
        scores = F.softplus(logits).sqrt()
        ids = torch.topk(scores+self.tensor('ffn.gate.bias'), c['num_experts_per_tok'], sorted=True).indices
        weights = scores.gather(-1, ids)
        weights /= weights.sum(-1, keepdim=True)+1e-20
        combined = torch.zeros_like(x, dtype=torch.float32)
        for expert_id, route_weight in zip(ids[0].tolist(), weights[0].tolist(), strict=True):
            combined += self.expert(x, f'experts.{expert_id}').float()*route_weight
        routed = combined.bfloat16()
        shared = self.expert(x, 'shared_experts')
        ffn = (shared.float()+routed.float()*c['routed_scaling_factor']).bfloat16()
        output = self.op('mhc_post', ffn, after, post, comb)
        return {'attention': attn, 'after_attention': after, 'ffn_input': x, 'shared_expert': shared,
            'routed_experts_unscaled': routed, 'ffn': ffn, 'output_trunk': output, 'next_pre_mix': next_pre,
            'topk_ids': ids, 'topk_weights': weights}


class CpuIntLinear(torch.nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('scale', scale)

    def forward(self, x):
        return cpu_linear(x, self.weight, self.scale)


class CpuFloatLinear(FloatLinear):
    """Explicit FP32 accumulation for the CPU golden integration only.

    The portable CPU BF16 GEMV reduction is not guaranteed to use the same
    accumulation as an Ascend BF16 matmul. Production NPU loading still uses
    FloatLinear with the stored BF16 weights.
    """
    def forward(self, x):
        return linear(x, self.weight)


class CpuIntExpert(torch.nn.Module):
    def __init__(self, w1, s1, w3, s3, w2, s2, *, limit):
        super().__init__()
        self.payload = [(w1, s1), (w3, s3), (w2, s2)]
        self.limit = limit

    def forward(self, x):
        return cpu_expert(x, self.payload, self.limit)


def compare(actual, expected):
    result = {}
    for name, reference in expected.items():
        value = actual[name].cpu()
        if not value.is_floating_point():
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
            result[name] = {'exact': True}
            continue
        error = value.float()-reference.float()
        relative = float(error.norm()/reference.float().norm().clamp_min(1e-15))
        torch.testing.assert_close(value, reference, rtol=.02, atol=.002)
        if relative > .015:
            raise AssertionError(f'{name}: relative L2 {relative}')
        result[name] = {'relative_l2': relative, 'max_abs_error': float(error.abs().max())}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['cpu', 'npu'], required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--mhc-reference-source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--token-id', type=int, default=17)
    args = p.parse_args()
    torch.set_num_threads(4)
    reader = ModelSlimCheckpoint(args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank = 0
    device = 'cpu'
    if args.mode == 'npu':
        import os
        import torch_npu  # noqa: F401
        import torch.distributed as dist
        rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.npu.set_device(rank)
        if int(os.environ.get('WORLD_SIZE', '1')) > 1:
            dist.init_process_group('hccl')
        device = f'npu:{rank}'
    else:
        assert not getattr(getattr(torch, 'npu', None), '_initialized', False)
    token_id = args.token_id+rank
    c = reader.config['text_config']
    trunk = reader.rows('embed.weight', [token_id])[:, None].expand(-1, c['hc_mult'], -1).contiguous()
    previous = torch.zeros(1, c['hc_mult'], 1)
    previous[:, 0] = 1.
    expected = GoldenLayer(reader, args.mhc_reference_source).forward(trunk, previous)
    module = importlib.import_module('deepinfer.model_executor.models.ascend_w8a8_layer')
    original_expert, original_loader = module.AscendW8A8Expert, reader.load_linear
    try:
        if args.mode == 'cpu':
            # Explicit test substitution only, never part of deployment.
            module.AscendW8A8Expert = CpuIntExpert
            def load_linear(prefix, device, **kwargs):
                weight, scale = reader.linear_payload(prefix)
                return CpuFloatLinear(weight) if scale is None else CpuIntLinear(weight, scale)
            reader.load_linear = load_linear
        layer = module.AscendW8A8SwaLayer(reader, device)
        _, _, stages = layer(trunk.to(device), previous.to(device), return_stages=True)
        # Retain raw tensors even when numerical acceptance fails.
        torch.save({'actual': {k: v.cpu() for k, v in stages.items()}, 'expected': expected,
                    'trunk': trunk, 'previous': previous}, args.output_dir/f'comparison-rank{rank}.pt')
        metrics = {}
        for name, reference in expected.items():
            value=stages[name].cpu(); error=value.float()-reference.float()
            metrics[name]={'max_abs':float(error.abs().max()),
                'relative_l2':float(error.norm()/reference.float().norm().clamp_min(1e-15)),
                'reference_absmax':float(reference.float().abs().max()),
                'not_close_count':int((~torch.isclose(value.float(),reference.float(),rtol=.02,atol=.002)).sum())}
        (args.output_dir/f'diagnostic-rank{rank}.json').write_text(json.dumps(metrics,indent=2))
        results = compare(stages, expected)
    finally:
        module.AscendW8A8Expert, reader.load_linear = original_expert, original_loader
    torch.save({'trunk': trunk, 'previous': previous, 'stages': expected, 'token_id': token_id}, args.output_dir/f'golden-rank{rank}.pt')
    result = {'status': 'pass', 'mode': args.mode, 'token_id': token_id, 'rank': rank,
        'results': results, 'checkpoint': args.model_dir.name,
        'boundary': ('CPU integer emulation and real complete layer wiring vs independent official mHC/integer golden; no NPU proof'
                     if args.mode == 'cpu' else 'Real checkpoint INT8 full layer, mixed BF16 mHC/attention, actual HCCL EP when multi-rank'),
        'npu_initialized': bool(getattr(getattr(torch, 'npu', None), '_initialized', False))}
    (args.output_dir/f'result-rank{rank}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if args.mode == 'npu':
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
