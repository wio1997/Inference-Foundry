#!/usr/bin/env python3
"""Offline recurrence of necessary prefill graph signature fields."""
import json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sources={
 'run184':root/'evidence/20260925_loop048_prefill/run184/phase/rank0.jsonl',
 'run186':root/'evidence/20260925_loop048_prefill/run186/phase/rank0.jsonl',
 'run188_A':root/'evidence/20260925_loop048_prefill/run188/forward/rank0_A.jsonl',
 'run188_A2':root/'evidence/20260925_loop048_prefill/run188/forward/rank0_A2.jsonl',
 'run200_A':root/'evidence/20260925_loop052_prefill_submission/run200/phase/A/rank0.jsonl',
 'run200_B':root/'evidence/20260925_loop052_prefill_submission/run200/phase/B/rank0.jsonl',
 'run200_A2':root/'evidence/20260925_loop052_prefill_submission/run200/phase/A2/rank0.jsonl',
}
cohorts={}
for k,p in sources.items():
 rows=[json.loads(z) for z in p.read_text().splitlines()]
 rows=[x for x in rows if x.get('mode')=='NONE' and x.get('num_reqs',0)>0]
 cohorts[k]=[(x['num_actual_tokens'],x['num_reqs']) for x in rows]
counts=collections.Counter(sig for seq in cohorts.values() for sig in seq)
all_count=sum(len(x) for x in cohorts.values())
result={'run':'run204','source_cohorts':{k:[{'actual_tokens':a,'num_reqs':n} for a,n in v] for k,v in cohorts.items()},'total_prefill_forwards':all_count,'unique_token_request_signatures':len(counts),'signature_counts':[{'actual_tokens':a,'num_reqs':n,'count':c} for (a,n),c in counts.most_common()],'repeated_signature_call_fraction':sum(c for c in counts.values() if c>1)/all_count,'most_common_signature_call_fraction':counts.most_common(1)[0][1]/all_count,'cross_service_signature_repeats':{str(sig):c for sig,c in counts.items() if c>=3},'limitations':['This is only (actual token count, request count), necessary but grossly insufficient for graph replay. Query lengths, block/slot tables, CP partitions, compression cardinalities, persistent tensor addresses and KV write set must also match or be refreshed.','Sources are separate services and one rank representative; warmed48 prefill calls were not traced, so capture amortization over formal workflow is unknown.','Run188 B admission-hold is excluded; Run200 B diagnostic same-stream still uses standard admission.','A graph with shape reuse may still violate correctness or cost more than eager due padding/capture/state restoration.']}
out=root/'evidence/20260925_loop053_native_prefill/run204/analysis.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('total_prefill_forwards','unique_token_request_signatures','repeated_signature_call_fraction','most_common_signature_call_fraction','signature_counts')},indent=2))
