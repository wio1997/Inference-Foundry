#!/usr/bin/env python3
"""Join six selected MC2-overlap waits to already extracted trace metadata."""
import hashlib,json
from decimal import Decimal
from pathlib import Path
p=Path(__file__).resolve().parent
files=[p/'ep8-mc2-selected-events.json',p/'ep8-trace-event-args.json']
mc2,trace=(json.loads(x.read_text()) for x in files)
rows=[]
for sample in mc2['samples']:
 for wait in sample['selected_events']:
  targets=[x for x in trace['targets'] if x['target'].get('run')==sample['run'] and x['target'].get('kind')=='EVENT WAIT' and x['target'].get('start_ns')==wait['start_ns'] and x['target'].get('end_ns')==wait['end_ns'] and x['target'].get('stream')==wait['stream'] and x['target'].get('global_task_id')==wait['global_task_id']]
  assert len(targets)==1 and not targets[0]['truncated']
  matches=[x for x in targets[0]['nearby_trace_events'] if x['name']=='EVENT WAIT' and x['phase']=='X' and x['tid']==wait['stream'] and x['numeric_identifier_args'].get('Model Id')==49 and x['numeric_identifier_args'].get('Task Id')==wait['task_id'] and Decimal(x['timestamp_us'])*1000==wait['start_ns']]
  assert len(matches)==1
  match=matches[0]
  rows.append({'run':sample['run'],'stream':wait['stream'],'task_id':wait['task_id'],'global_task_id':wait['global_task_id'],'trace_id':match['id'],'trace_id2':match['id2'],'filtered_identifier_args':match['numeric_identifier_args'],'filtered_relevant_arg_keys':match['relevant_arg_keys']})
assert len(rows)==6
print(json.dumps({'input_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files},'trace_sha256_recorded':trace['trace_sha256'],'scope':'six selected waits; prior export has filtered argument keys, so unexported original arguments remain unknown','rows':rows},indent=2))
