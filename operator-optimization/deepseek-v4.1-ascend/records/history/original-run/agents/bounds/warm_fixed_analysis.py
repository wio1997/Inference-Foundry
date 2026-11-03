#!/usr/bin/env python3
"""Read saved metadata only and reproduce conditional residence arithmetic."""
import hashlib
import json
import re
from pathlib import Path

here = Path(__file__).resolve().parent
run = here.parent.parent
project = run.parents[4]
config = run / 'evidence/model-config.json'
model = run / 'sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py'
index = project / 'source172/model-metadata/quant_model_weights.safetensors.index.json'
description = project / 'source172/model-metadata/quant_model_description.json'
index_map = json.loads(index.read_text())['weight_map']
desc = json.loads(description.read_text())
cfg = json.loads(config.read_text())['text_config']
keys = sorted([k for k in index_map if re.fullmatch(r'layers\.\d+\.attn\.wq_b\.weight', k)], key=lambda k: int(k.split('.')[1]))
ids = [int(k.split('.')[1]) for k in keys]
assert ids == list(range(cfg['num_hidden_layers'])) == list(range(40))
assert all(desc.get(k) == 'W8A8_DYNAMIC' for k in keys)
k = cfg['q_lora_rank']
n = cfg['num_attention_heads'] * cfg['head_dim']
one, layers = k*n, cfg['num_hidden_layers']
total = one*layers
example_capacity = 192 * 1024 * 1024
files = [config,model,index,description]
out = {
    'status': 'CONDITIONAL_RESIDENCY_BOUND_NOT_PROVEN_FOR_ACTUAL_RUN',
    'config': {'layers':layers,'K':k,'N':n},
    'local_index_40_unique_parameter_names': len(keys),
    'local_description_all_dynamic_w8a8':True,
    'index_does_not_prove_distinct_device_storage_or_distinct_values': True,
    'individual_raw_int8_matrix_bytes_if_full_and_replicated':one,
    'forty_raw_int8_matrices_if_distinct_full_and_replicated_bytes':total,
    'fast_storage_capacity_example_bytes_assumed_all_inclusive_not_hardware_fact':example_capacity,
    'required_transfer_bytes_under_explicit_all_byte_use_and_storage_assumptions':max(0,total-example_capacity),
    'physical_npu_or_hbm_bandwidth_upper_proven':False,
    'target_actual_execution_satisfies_assumptions_proven':False,
    'formal_end_to_end_tokens_per_second_ceiling':'UNKNOWN',
    'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
}
(here/'warm_fixed_analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
