"""Stop exact owned model/proxy after verified final snapshot, on user pause."""
from pathlib import Path
import json,socket,subprocess,urllib.request
node={'B4-33U-Atlas900RCKA3-32':32,'B5-5U-Atlas900RCKA3-33':33}[socket.gethostname()]
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/'+('indexer-k-graph-live1-20261019/node32' if node==32 else 'indexer-k-graph-live2-20261019/node33'))
cid={32:'70e58ecd72b4bd71c4ed0b1e5874c6fa0ee6413ee4d661d5ffe701acde579b96',33:'a4db690c1cf2ae583cdcbf84464e1393a24ab4dd3f9df4b192be64c3e2777011'}[node]
plan=json.loads((root/'plan.json').read_text());snapshot=json.loads((root/'indexer-k-graph-final-snapshot-v1.json').read_text())
assert snapshot['completed'] and snapshot['node']==node
assert snapshot['snapshots'][0]['CID']==cid
items=[('model',cid)]
if node==32:
 proxy=json.loads((root/'official-proxy-v1/launch.json').read_text())
 assert proxy['CID']=='29a243147533f7b27bee4cb1c99d55c811099b884ac91e233e2d01428dc10114' and proxy['parent_P_CID']==cid
 items.insert(0,('official-proxy',proxy['CID']))
else:
 assert not(root/'indexer-k-graph-overlap-probe/enable').exists()
 assert not(root/'indexer-k-graph-overlap-probe/profile-copy-labels').exists()
rows=[]
for kind,ident in items:
 before=json.loads(subprocess.check_output(['docker','inspect',ident]))[0]
 assert before['Image']==plan['image']['id'] and before['Id']==ident
 if kind=='model':
  assert before['Name']=='/'+plan['container']['name'] and ident==json.loads((root/'launch.json').read_text())['container_id']
  if before['State']['Running']:
   with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(node,29481 if node==32 else 29482),timeout=5) as f:data=f.read().decode()
   lines=[x for x in data.splitlines() if x.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))]
   assert len(lines)>=3 and all(float(x.rsplit(' ',1)[1])==0 for x in lines)
 elif kind=='official-proxy':
  assert not before['HostConfig'].get('Devices')
 if before['State']['Running']:
  subprocess.run(['docker','stop','--time','30',ident],check=True,capture_output=True,timeout=50)
 after=json.loads(subprocess.check_output(['docker','inspect',ident]))[0]
 assert not after['State']['Running']
 rows.append(dict(kind=kind,CID=ident,before=before['State'],after=after['State']))
report=dict(completed=True,node=node,rows=rows,scope='User-requested pause after final snapshot backup. Exact owned CIDs only; no deletion, restart or other workloads touched.')
(root/'indexer-k-graph-final-stop-v1.json').write_text(json.dumps(report,indent=2)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
