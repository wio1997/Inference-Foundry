"""Recheck saved CPU-only evidence, with explicit dispatch-key conservation."""
import json
from pathlib import Path
p=Path(__file__).with_name('CPU_result_A.json');d=json.loads(p.read_text());assert d['cases']==36 and len(d['states'])==36
assert d['model_requests']==d['NPU_operations']==0 and not any(d['device_state_before'].values()) and d['device_state_before']==d['device_state_after']
for row in d['states']:
 states=row['states'];assert len(states)==5;before=states[0]
 for s in states:
  assert all(s[k]==before[k] for k in ['inference_mode','grad_enabled','included','excluded'])
 assert row['output_aliases_input']
 if row['mode']=='inference':assert row['output_is_inference']==(row['input']=='inference')
print('36 saved cases: inference/grad/TLS/alias contracts conserved; CPU only')
