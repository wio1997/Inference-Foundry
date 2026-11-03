#!/usr/bin/env python3
"""CPU-only extraction from one saved 169 service log and fixed local schema summary."""
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parents[2]
P=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/raw-results/service-log-tail.txt')
raw=P.read_bytes();lines=raw.decode('utf-8').split('\n')
patterns={'cache_group_sizes':r'kv cache group sizes \[([^\]]+)\]',
          'kv_cache_groups_literal':r'kv_cache_groups', 'draft_attn_groups_literal':r'draft_attn_groups',
          'dynamic_spec_config_literal':r'dynamic_spec_config', 'spec_five':r"method='dspark'.*?num_spec_tokens=5"}
out={}
for name,pattern in patterns.items():
 vals=[]
 for i,line in enumerate(lines,1):
  m=re.search(pattern,line,re.IGNORECASE)
  if m:
   item={'line':i}
   if name=='cache_group_sizes':item['values']=[int(x.strip()) for x in m.group(1).split(',')]
   vals.append(item)
 out[name]=vals
schema_rel='evidence/trace-schema-inspection.json';schema_path=R/schema_rel;schema=json.loads(schema_path.read_text())
keep=[]
for name,sql in schema['tables']:
 if name in ('TASK','COMPUTE_TASK_INFO','PYTORCH_API','STRING_IDS'):
  keep.append({'name':name,'schema':sql})
result={'log':str(P),'log_sha256':hashlib.sha256(raw).hexdigest(),'lines':len(lines),'matches':out,
        'trace_schema_summary_path':schema_rel,'trace_schema_summary_sha256':hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        'trace_relevant_table_schemas':keep,
        'limits':['Saved log tail only; no match does not prove absence in unrecorded logs or other servers.',
                  'KV group-size list includes engine groups, not a mapping to DSpark draft attention groups.',
                  'Profiler shape schema does not provide a tensor storage identity or group-to-layer mapping.',
                  'No device, service, or remote access used.']}
print(json.dumps(result,indent=2))
