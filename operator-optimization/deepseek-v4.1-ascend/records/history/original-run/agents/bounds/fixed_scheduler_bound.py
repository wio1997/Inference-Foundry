#!/usr/bin/env python3
"""Reproduce conditional finite-batch counts; never access network or device."""
import hashlib
import json
from pathlib import Path

p=Path(__file__).resolve().parent
run=p.parent.parent
sources={
 'launch':run/'evidence/frozen-serve.sh',
 'sampler_source_excerpt':p/'sampler-fixed-commit-web.txt',
 'scheduler_source_excerpt':p/'scheduler-fixed-commit-web.txt',
 'benchmark':run/'evidence/fixed-output-conditional-bound.json',
 'dmi':p/'stage-dmi-readonly.json',
 'oem_catalog_search':p/'isoftstone-catalog-search.txt',
}
launch=sources['launch'].read_text();sampler=sources['sampler_source_excerpt'].read_text();scheduler=sources['scheduler_source_excerpt'].read_text()
assert '--max-num-seqs 4' in launch and '--data-parallel-size 2' in launch
assert '"method":"dspark","num_speculative_tokens":5' in launch
assert '(batch_size, max_spec_len + 1)' in sampler
assert 'Maximum number of sequences to be processed in a single iteration' in scheduler
Q=8*1024; S=4; G=5+1; E=2
events=(Q+S*G-1)//(S*G)
per_engine=(events+E-1)//E
out={
 'status':'CONDITIONAL_FIXED_IMPLEMENTATION_EVENT_COUNT_NOT_PHYSICAL_UNIVERSAL_CEILING',
 'assumed_token_count':Q,'assumed_engines':E,'conditional_sequences_per_iteration':S,
 'assumed_max_committed_tokens_per_sequence_per_iteration':G,
 'minimum_total_engine_commit_events_if_assumptions_hold':events,
 'minimum_commit_events_for_at_least_one_engine':per_engine,
 'equal_min_spacing_conditional_lower_seconds':'170 * delta_seconds',
 'equal_min_spacing_conditional_upper_tps':'8192 / (170 * delta_seconds)',
 'interval_startup_and_scope':'All 8192 output tokens assigned to events inside timing interval; per-DP event timestamps separated by at least delta; delta must be justified over all legal schedules in frozen domain.',
 'same_commit_web_source_is_running_file':'UNVERIFIED',
 'runtime_sampler_and_metadata_output_identity':'UNVERIFIED',
 'conditional_max_6_applies_all_outputs_including_prefill':'UNVERIFIED',
 'delta_hard_lower_seconds':None,'numerical_tps_upper':'UNKNOWN',
 'sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sources.items()},
 'no_npu_or_network_access':True,
}
(p/'fixed-scheduler-bound.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(out,indent=2,ensure_ascii=False))
