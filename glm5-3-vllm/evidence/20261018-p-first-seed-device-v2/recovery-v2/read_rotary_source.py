from pathlib import Path
import hashlib,json,subprocess
cid='084035e84399bd88c3788c80f184590f309fd6730f93cc308f460487c75af213';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
path='/vllm-workspace/vllm-ascend/vllm_ascend/ops/rotary_embedding.py';b=subprocess.check_output(['docker','exec',cid,'cat',path],timeout=20);out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False);(out/'rotary_embedding.py').write_bytes(b)
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'source_path':path,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'source':b.decode(),'scope':'Pinned live image source only; no imports, forward, profile, NPU allocation or model change.'})+'\nGLM53_RESULT_JSON_END')
