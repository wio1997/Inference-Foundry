from pathlib import Path
import hashlib
import json
import subprocess

cid='9774dd714ad98295848b710f3a08d25cf91203eba5a801e554d10060f67b594d'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Name']=='/glm53-mtp-moe-graph-33-live1-20261018'
assert d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False);rows=[]
for path,functions in [
 ('/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py', ['def _moe_forward(', 'def _moe_forward_shared(']),
 ('/vllm-workspace/vllm/vllm/v1/spec_decode/llm_base_proposer.py', ['    def _create_draft_vllm_config(']),
]:
    b=subprocess.check_output(['docker','exec',cid,'cat',path],timeout=20)
    assert len(b)<1024**2
    (out/Path(path).name).write_bytes(b);text=b.decode();snippets=[]
    for function in functions:
        begin=text.index(function);delimiter='\n    def ' if function.startswith('    ') else '\ndef '
        end=text.find(delimiter,begin+len(function));snippets.append(text[begin:end if end>=0 else len(text)])
    rows.append({'path':path,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'snippets':snippets})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'CID':cid,'sources':rows,'compute_contexts_created':False,'scope':'Pinned-source CPU read: draft inherits compilation config, and native opaque MoE op dynamically resolves layer._forward_impl. No weights/model imported or GPU operation.'})+'\nGLM53_RESULT_JSON_END')
