from pathlib import Path
import json,subprocess
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/sfa-replay-order-live1-20261018/node33');cid='e202811aa3535d277385bb4a401ada4cf0c6fcfec5536a3adae2b6724d53210f';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Id']==cid and d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
rows=[]
for p in sorted((root/'sfa-replay-order-probe').glob('rank-*.jsonl')):
 for line in p.read_text().splitlines():
  z=json.loads(line);z['source']=p.name;rows.append(z)
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'CID':cid,'rows':rows,'ranks':sorted(set(x['tp_rank'] for x in rows)),'marker_present':(root/'sfa-replay-order-probe/enable').exists()})+'\nGLM53_RESULT_JSON_END')
