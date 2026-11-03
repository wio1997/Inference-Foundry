#!/usr/bin/env python3
"""Inventory frozen source stages and the local profiler schema; no remote IO."""
import hashlib,json,re
from pathlib import Path
root=Path(__file__).resolve().parents[2]
src=root/'sources/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp'
inv=root/'evidence/profiler-export-inventory.json'
sample=root/'evidence/trace-kernel-inspection.json'
out=Path(__file__).with_name('mc2-stage-field-audit.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lines=src.read_text().splitlines()
terms=['GMM1(params);','GMM2(params);','DispatchAndCombine(params);','while (flag < flagBase)', 'CrossCoreWaitFlag<0x2>(SYNCFLAGV2C)', 'CrossCoreWaitFlag<0x2>(SYNCFLAGC2V)']
positions={term:[i for i,s in enumerate(lines,1) if term in s] for term in terms}
assert all(positions.values())
examples=json.loads(inv.read_text())['examples']
columns={k:v['columns'] for k,v in examples.items() if isinstance(v,dict) and 'columns' in v}
kernel=columns['ASCEND_PROFILER_OUTPUT/kernel_details.csv']
phase=[s for s in kernel if re.search(r'(dispatch|combine|gmm1|gmm2|readiness|phase)',s,re.I)]
ex=json.loads(sample.read_text())['cube_low']
mc2=[x for x in ex if x.get('Type')=='DispatchFFNCombine']
assert mc2 and not phase
result={'local_evidence_scope':'frozen source, local rank0 export schema example and one kernel sample; not a complete query of all original remote traces',
 'sha256':{str(p.relative_to(root)):sha(p) for p in (src,inv,sample)},'source_stage_marker_lines':positions,
 'kernel_details_columns':kernel,'explicit_stage_columns':phase,'recorded_mc2_example':mc2[0],
 'assessment':{'source_stage_control_flow':'SUPPORTED; these markers show programmed stages and waits, not their observed durations',
 'explicit_mc2_phase_event_in_local_example':'NOT_OBSERVED',
 'internal_readiness_wait_duration':'UNKNOWN; source has flag waits, exported aggregate fields inspected do not timestamp those flags',
 'dispatch_gmm_combine_time':'UNKNOWN; task duration and whole-kernel pipeline ratios do not identify stages',
 'kernel_wait_time_column':'present in schema; semantics and whether it measures internal flag waits not established here; do not use as such',
 'all_remote_trace_no_stage_data':'UNKNOWN; this audit is restricted to local examples and does not prove absence in all remote raw files'},
 'next_action':'Read raw per-event schema and MC2-related event IDs in authorized 169 trace, or capture documented internal markers if safely available; require mapping to exact kernel, ranks, and per-phase boundaries.'}
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'markers':positions,'kernel_column_count':len(kernel),'explicit_stage_columns':phase,'mc2_sample_count':len(mc2),'out':str(out)},indent=2))
