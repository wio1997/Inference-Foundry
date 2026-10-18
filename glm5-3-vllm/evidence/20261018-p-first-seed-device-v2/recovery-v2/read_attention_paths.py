from pathlib import Path
import json,subprocess
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefill-attention-live1-20261018/node32')
cid='8df6987f0793978af501a754b5b186a120cf4255169f5eed8835a5defb8b3148'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==cid
rows=[]
for p in sorted((root/'prefill-attention-probe').glob('rank-*.jsonl')):
 for line in p.read_text().splitlines():
  row=json.loads(line);row['source']=p.name;rows.append(row)
report={'CID':cid,'rows':rows,'ranks':sorted(set(row['tp_rank'] for row in rows)),
 'marker_present':(root/'prefill-attention-probe/enable').exists()}
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
