#!/usr/bin/env python3
"""Recover compact evidence from complete owned diagnostic files; zero model calls."""
import hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('b',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0780-long-diagnostic-artifacts-v1'
HOST=r'''
import socket,pathlib,json,hashlib,struct
assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T0710-long-diagnostic-v1');out={}
for name in ['diagnostic-result.json','diagnostic-progress.json']:
 p=root/name;raw=p.read_bytes();data=json.loads(raw)
 for case in data['cases']:
  for row in case['rows']:
   for choice in row.get('response',{}).get('choices',[]):
    ids=choice.pop('prompt_token_ids',None)
    if ids is not None:
     digest=hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()
     assert len(ids)==80000 and digest=='43a309afc4f0652d712dfa3c5318735b680a685ba179d49539c9a1d952a98529'
     choice['prompt_token_ids_evidence']={'count':len(ids),'sha256_le_u32':digest,'complete_exact_input_verified':True}
 out[name]={'path':str(p),'raw_file_bytes':len(raw),'raw_file_sha256':hashlib.sha256(raw).hexdigest(),'compact_data':data}
encoded=json.dumps(out);assert len(encoded.encode())<200000
print(encoded)
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,hostname=b.NODES[43];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.43'
 cmd=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 r=b.invoke([cmd],HOST,timeout=90);(RUN/'node-43.json').write_text(json.dumps(r,indent=2)+'\n')
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':r['rc'],'model_requests':0},indent=2)+'\n');print('artifact recovery',r['rc'],flush=True)
if __name__=='__main__':main()
