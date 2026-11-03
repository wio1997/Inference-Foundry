#!/usr/bin/env python3
"""Recompute arithmetic only, no device access; proof/conditions: cold_gemm_bound.md."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
run = here.parent.parent
sources = ['bounds/Wq_b_rank0_M3.input.json', 'calculations/Wq_b_rank0_M3.json',
           'evidence/conditional-performance-comparisons.json', 'evidence/IDEAL-MODEL.md']
hashes = {p: hashlib.sha256((run / p).read_bytes()).hexdigest() for p in sources}
m, k, n = 3, 1280, 32768
weight_bytes, activation_bytes = k*n, m*k
macs = m*k*n
conditional_cube_cap = 353_894_400_000_000
conditional_compute_time = Fraction(macs, conditional_cube_cap)
result = {
    'label': 'HYPOTHETICAL_COLD_DENSE_INT8_EXACT_INT32_INPUT_BOUND_ONLY',
    'scope': 'All-ones input instance or worst-case time for independent cold variable-weight exact int32 GEMM; specified raw input at HBM; full outputs required; no compressed shadow/precomputation. Not a bound for every input or average time; not established for measured Wq_b or end-to-end inference.',
    'shape': {'M':m,'K':k,'N':n},
    'raw_weight_bytes':weight_bytes,
    'raw_input_bytes':activation_bytes,
    'necessary_read_bytes_worst_case':weight_bytes+activation_bytes,
    'necessary_read_expression_seconds':'41946880 / B_HBM_read_max_Bps',
    'conditional_compute_time_fraction_seconds':str(conditional_compute_time),
    'conditional_compute_time_seconds':float(conditional_compute_time),
    'read_bandwidth_threshold_for_dominance_Bps':str(Fraction(weight_bytes+activation_bytes,1)/conditional_compute_time),
    'combined_lower_expression':'max(125829120 / P_MAC_max, 41946880 / B_HBM_read_max_Bps)',
    'conditional_row_throughput_upper_expression':'3 / max(125829120 / P_MAC_max, 41946880 / B_HBM_read_max_Bps)',
    'actual_hot_wqb_bound_status':'UNKNOWN: applicability of cold state, exact arithmetic contract and physical max bandwidth not established',
    'hard_bandwidth_upper_Bps':None,
    'target_code_or_hardware_modified':False,
    'input_sha256':hashes,
}
(here/'cold_gemm_bound.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,indent=2,ensure_ascii=False))
