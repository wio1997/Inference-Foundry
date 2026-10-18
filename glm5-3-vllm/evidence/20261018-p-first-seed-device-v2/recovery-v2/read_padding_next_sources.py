from pathlib import Path
import hashlib,json,subprocess
cid='28e687e363317e918ed636a4615f918dda3ea80839626e357e82d77ea87faf56'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
paths=['/vllm-workspace/vllm-ascend/vllm_ascend/attention/sfa_v1.py','/vllm-workspace/vllm-ascend/vllm_ascend/compilation/acl_graph.py']
rows=[]
for path in paths:
 b=subprocess.check_output(['docker','exec',cid,'cat',path],timeout=20);rows.append({'path':path,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'CID':cid,'source_hashes':rows})+'\nGLM53_RESULT_JSON_END')
