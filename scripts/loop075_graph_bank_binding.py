#!/usr/bin/env python3
"""Diagnostic: distinct packed bank values must survive GMM2 graph capture."""
import json
import argparse
from pathlib import Path

import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
torch.npu.config.allow_internal_format = True
rank = 4
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int, required=True)
parser.add_argument('--banks', type=int, choices=(2, 8), default=2)
args = parser.parse_args()
torch.npu.set_device(rank)
root = Path('/data/wio/Inference_Foundry')
out = root / f'evidence/20260927_loop075_bound/run{args.run}'
out.mkdir(parents=True, exist_ok=True)
snap = json.loads((root / f'evidence/20260925_loop039_gmm/run121/counts/rank{rank}_cycle64.json').read_text())
counts = snap['rows'][64]['counts']
group = torch.tensor(counts, device=f'npu:{rank}', dtype=torch.int64)
x = torch.ones((576, 2048), device=f'npu:{rank}', dtype=torch.int8)
values = [int(str(i) * 8, 16) for i in range(1, args.banks + 1)]
weights = [torch_npu.npu_format_cast(torch.full((32, 2048, 512), value if value < 2**31 else value - 2**32,
                                                 device=f'npu:{rank}', dtype=torch.int32), 29)
           for value in values]
assert len({w.data_ptr() for w in weights}) == args.banks
unit_quant = torch_npu.npu_trans_quant_param(
    torch.ones((1, 4096), device=f'npu:{rank}', dtype=torch.float32))
scale = unit_quant.unsqueeze(0).expand(32, 1, 4096).contiguous()
per_token_scale = torch.ones((576,), device=f'npu:{rank}', dtype=torch.float32)


def call(i):
    return torch_npu.npu_grouped_matmul(x=[x], weight=[weights[i]], scale=[scale], bias=None,
                                        per_token_scale=[per_token_scale], split_item=2,
                                        group_list_type=1, group_type=0, group_list=group,
                                        output_dtype=torch.bfloat16)[0]


eager = [call(i) for i in range(args.banks)]
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    captured = [call(i % args.banks) for i in range(16)]
graph.replay()
torch.npu.synchronize()
summary = {
    'status': 'diagnostic', 'rank': rank, 'ordinal': 64, 'banks': args.banks,
    'distinct_weight_ptr': True,
    'eager_bank0_first': eager[0][0, 0].item(),
    'eager_bank1_first': eager[1][0, 0].item(),
    'graph_bank0_first': captured[0][0, 0].item(),
    'graph_bank1_first': captured[1][0, 0].item(),
    'eager_max_abs_difference': (eager[0].float() - eager[1].float()).abs().max().item(),
    'graph_max_abs_difference': (captured[0].float() - captured[1].float()).abs().max().item(),
    'graph_even_against_bank0': (captured[0].float() - eager[0].float()).abs().max().item(),
    'graph_odd_against_bank1': (captured[1].float() - eager[1].float()).abs().max().item(),
    'eager_first_values': [t[0, 0].item() for t in eager],
    'graph_first_values': [t[0, 0].item() for t in captured],
    'all_bank_replay_match': all(bool(torch.equal(captured[i], eager[i % args.banks])) for i in range(16)),
    'all_bank_active_prefix_match': all(bool(torch.equal(captured[i][:sum(counts)], eager[i % args.banks][:sum(counts)])) for i in range(16)),
    'active_prefix_max_abs_diff': max((captured[i][:sum(counts)].float() - eager[i % args.banks][:sum(counts)].float()).abs().max().item() for i in range(16)),
    'all_graph_outputs_finite': all(bool(torch.isfinite(t).all().item()) for t in captured),
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
