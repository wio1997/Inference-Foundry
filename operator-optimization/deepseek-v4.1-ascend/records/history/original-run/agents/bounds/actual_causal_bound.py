#!/usr/bin/env python3
"""Reproduce scoped output-count arithmetic and save evidence hashes."""
import hashlib
import json
import math
from pathlib import Path

here=Path(__file__).resolve().parent
run=here.parent.parent
paths={
  'launch':run/'evidence/frozen-serve.sh',
  'benchmark':run/'evidence/fixed-output-conditional-bound.json',
  'proposer':run/'sources/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py',
  'draft_loop':run/'sources/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py',
  'proposer_buffer':run/'sources/vllm-workspace/vllm/vllm/v1/spec_decode/llm_base_proposer.py',
  'capacity':run/'evidence/conditional-nominal-capacity.json',
}
launch=paths['launch'].read_text()
assert '"num_speculative_tokens":5' in launch and '"method":"dspark"' in launch
outputs=1024
conditional_max_per_group=6
out={
 'frozen_launch_speculative_method':'dspark',
 'frozen_launch_max_draft_tokens':5,
 'conditional_one_request_output_tokens':outputs,
 'assumed_max_committed_tokens_per_group':conditional_max_per_group,
 'conditional_min_commit_groups_per_request':math.ceil(outputs/conditional_max_per_group),
 'conditional_min_request_groups_for_8_requests':8*math.ceil(outputs/conditional_max_per_group),
 'group_definition_and_limit_scope':'Must include prefill commits; all outputs assigned once; actual sampler commit bound unproven; can batch groups across requests; cannot infer duration or serialized target launches from count.',
 'baseline_nominal_capacity_status':'CONDITIONAL; 1800MHz profiler metadata is not proof of hard max; 9382 formula applicability and joint resource capacity not proven',
 'unconditional_end_to_end_tps_upper':'UNKNOWN',
 'no_device_actions':True,
 'source_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
}
(here/'actual_causal_bound.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(out,indent=2,ensure_ascii=False))
