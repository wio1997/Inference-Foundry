from pathlib import Path
import json,subprocess
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-moe-graph-live1-20261018/node33');cid='9774dd714ad98295848b710f3a08d25cf91203eba5a801e554d10060f67b594d';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Id']==cid and d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
rows=[]
for p in sorted((root/'mtp-moe-graph-probe').glob('rank-*.jsonl')):
 for line in p.read_text().splitlines():
  z=json.loads(line);z['source']=p.name;rows.append(z)
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'CID':cid,'rows':rows,'ranks':sorted(set(x['rank'] for x in rows)),'marker_present':(root/'mtp-moe-graph-probe/enable').exists()})+'\nGLM53_RESULT_JSON_END')
