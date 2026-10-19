from pathlib import Path
import json,subprocess
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-ffn-graph-live2-20261018/node33');cid='7b65366a6d14d7840e2686118bb29e3621d470424e30f2648f5eb237a395906d';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Id']==cid and d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
rows=[]
for p in sorted((root/'mtp-ffn-graph-probe').glob('rank-*.jsonl')):
 for line in p.read_text().splitlines():
  z=json.loads(line);z['source']=p.name;rows.append(z)
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'CID':cid,'rows':rows,'ranks':sorted(set(x['rank'] for x in rows)),'marker_present':(root/'mtp-ffn-graph-probe/enable').exists()})+'\nGLM53_RESULT_JSON_END')
