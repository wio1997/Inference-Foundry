from pathlib import Path
import hashlib,json,socket,subprocess
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
cid='9b788715cb6e34139c2e9a70ef5bcf49009dda507164bf11e9e00cd07c163863'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Name']=='/glm53-early-copy-shadow-33-live1-20261019'
assert d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False);rows=[]
for rel in ['vllm/vllm/v1/outputs.py','vllm/vllm/config/vllm.py']:
    b=subprocess.check_output(['docker','exec',cid,'cat','/vllm-workspace/'+rel],timeout=20)
    p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    rows.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(dict(completed=True,files=rows,scope='Stdlib-only exact-image schema/config source read. No model imports, NPU work or changes.'))+'\nGLM53_RESULT_JSON_END')
