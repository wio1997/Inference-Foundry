from pathlib import Path
import json,subprocess
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefill-padding-live1-20261018/node32')
cid='28e687e363317e918ed636a4615f918dda3ea80839626e357e82d77ea87faf56'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==cid
rows=[]
for p in sorted((root/'prefill-padding-probe').glob('rank-*.jsonl')):
 for line in p.read_text().splitlines():
  row=json.loads(line);row['source']=p.name;rows.append(row)
report={'CID':cid,'rows':rows,'ranks':sorted(set(row['tp_rank'] for row in rows)),
 'marker_present':(root/'prefill-padding-probe/enable').exists()}
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
