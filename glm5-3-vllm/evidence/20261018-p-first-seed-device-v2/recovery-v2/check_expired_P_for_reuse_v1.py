from pathlib import Path
import json,socket,subprocess,time
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-publication-live1-20261019/node32')
plan=json.loads((root/'plan.json').read_text());cid='84bdaf268d006b2d4d172bc6f9adb8f5f74cc9f38e1a622644536269f0918f8b'
proxy=json.loads((root/'official-proxy-v1/launch.json').read_text());assert proxy['parent_P_CID']==cid and proxy['CID']=='a1fec8aedbff1f5fdbbba0e1d1ee40b21d9aecd52a31a0e9a80f79e01f8e2ef3'
rows=[]
for name,ident in [('model',cid),('proxy',proxy['CID'])]:
 for i in range(10):
  d=json.loads(subprocess.check_output(['docker','inspect',ident]))[0]
  assert d['Id']==ident and d['Image']==plan['image']['id']
  if not d['State']['Running']:break
  time.sleep(3)
 assert not d['State']['Running'] and not d['State']['OOMKilled'] and d['State']['ExitCode']==124
 rows.append(dict(kind=name,CID=ident,state=d['State']))
s=subprocess.check_output(['npu-smi','info'],text=True,timeout=30);assert s.count('No running processes found in NPU')==8
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(dict(completed=True,old_exact_P_proxy_watchdogs_expired=True,states=rows,empty_physical_cards=8,no_stop_issued=True))+'\nGLM53_RESULT_JSON_END')
