#!/usr/bin/env python3
"""Read-only source/evidence identity gate for one production TP dependency slice.

This deliberately emits no timing or Bound endpoint.  A live selected-Graph
branch and typed producer/consumer witness are separate admission steps.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
ASC = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
HIST = Path('/data/wio/performance_history/dsv4f-w4a8-ascend-logs')
OUT = ROOT / 'evidence/20260928_loop080_bound/run581/source_gate.json'

SOURCES = {
    'dsa_cp': ASC / 'attention/context_parallel/dsa_cp.py',
    'linear_op': ASC / 'ops/linear_op.py',
    'model': ASC / 'models/deepseek_v4.py',
    'forward_context': ASC / 'ascend_forward_context.py',
    'serve': ROOT / 'scripts/serve.sh',
    'run378_ledger': ROOT / 'evidence/20260927_loop077_bound/run378/ledger.json',
    'run502_findings': ROOT / 'evidence/20260927_loop079_identity/run502/findings.md',
    'run508_findings': ROOT / 'evidence/20260927_loop079_identity/run508/findings.md',
    'run510_review': ROOT / 'evidence/20260927_loop079_identity/run510/astra_terminal_abi_source_review.md',
    'run576_findings': ROOT / 'evidence/20260928_loop080_bound/run576/findings.md',
    'run580_findings': ROOT / 'evidence/20260928_loop080_bound/run580/findings.md',
    'r20_report': HIST / 'rounds/R20_dsa_cp_prefill_tp_ag_q_overlap/REPORT.md',
    'r27_report': HIST / 'rounds/R27_post_keep_prefill_device_critical_path/REPORT.md',
}
PREDICATES = {
    'conditional_dsa_wo_b_call': ('dsa_cp', 'return self.wo_b(o_proj_input)'),
    'attention_output_copy': ('dsa_cp', 'output[...] = self._apply_wo_b(o_proj_input, full_gather_wo_a_enabled)'),
    'row_parallel_class_selector': ('linear_op', 'return SequenceRowParallelOp(layer)'),
    'alternate_otp_selector': ('linear_op', 'return DSV4OProjRowParallelOp(layer)'),
    'conditional_partial_producer': ('linear_op', 'output_parallel = self.layer.quant_method.apply(self.layer, x, bias=bias_)'),
    'conditional_tp_rs': ('linear_op', 'output = tensor_model_parallel_reduce_scatter(output_parallel, 0)'),
    'attention_to_hc_post': ('model', 'hidden_states = self.hc_post(hidden_states, residual, post, comb)'),
    'tp_size_mmrs_gate': ('forward_context', 'mmrs_fusion = tp_world_size <= 8'),
    'serve_flashcomm_default': ('serve', 'FLASHCOMM1_ENABLED="${FLASHCOMM1_ENABLED:-true}"'),
}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    text = {key: path.read_text() for key, path in SOURCES.items()}
    positions = {}
    for name, (key, needle) in PREDICATES.items():
        lines = [i for i, line in enumerate(text[key].splitlines(), 1) if needle in line]
        if not lines:
            raise ValueError(f'{name}: missing source predicate')
        positions[name] = dict(source=key, lines=lines, needle=needle)
    ledger = json.loads(text['run378_ledger'])
    rows = ledger['ordered_tasks']
    counts = {kind: sum(x['kind'] == kind for x in rows)
              for kind in ('hcom_reduceScatter', 'hcom_allGather', 'hcom_alltoall')}
    if (len(rows), counts) != (265, {'hcom_reduceScatter': 87, 'hcom_allGather': 135, 'hcom_alltoall': 43}):
        raise ValueError('Run378 ledger signature drift')
    data = {
        'status': 'source_and_prior_gate_only',
        'contract': 'frozen DSpark7 acceptance/cycles/output, DP1TP8 FULL96, no algorithm intervention',
        'slice': 'layer0 DSA wo_b partial -> selected TP collective -> attention output copy -> first hc_post',
        'source_manifest': {key: {'path': str(path), 'sha256': sha(path)} for key, path in SOURCES.items()},
        'predicates': positions,
        'run378_inventory': {'total': len(rows), 'kinds': counts, 'candidate_ordinal_3': rows[3]},
        'certified_now': [
            'conditional source dependencies and alternative branch selectors',
            'Run378 current-order all8 HCCL call-kind/size inventory',
            'Run508 selected MLA update zip loop zero, separate from this slice',
        ],
        'not_certified': [
            'loaded live layer0 wo_b custom_op class and quant method',
            'selected sequence row versus alternate OTP and MMRS branch',
            'ordinal3 identity to this live layer0 output',
            'typed last writer, collective input/output, copy destination and hc_post consumer',
            'producer-ready, collective completion and consumer-ready timestamps',
            'same-W0 all8 service, legal overlap window or finite Bound',
        ],
        'strict_bounds': {'resource_hardware': None, 'scheduling_execution': None, 'product_e2e': None},
        'formal_current_tps': 571.681,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': data['status'], 'source_count': len(SOURCES),
                      'predicate_count': len(positions), 'ledger_counts': counts,
                      'output': str(OUT)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
