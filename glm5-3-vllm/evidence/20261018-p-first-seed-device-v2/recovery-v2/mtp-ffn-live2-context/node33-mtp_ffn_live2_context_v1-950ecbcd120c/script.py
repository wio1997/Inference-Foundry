from pathlib import Path
import hashlib,json,subprocess
cid='05e98c8425d8bb0db0c30425ae054f9e8ee7b747914a18defaa2f18d8aee8f2a';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
code="from pathlib import Path;import json;root=Path('/vllm-workspace/vllm-ascend/vllm_ascend');rows=[{'path':str(p),'line':i+1,'text':line.strip()} for p in root.rglob('*.py') for i,line in enumerate(p.read_text().splitlines()) if 'moe_layer_index' in line];print(json.dumps(rows))"
info=json.loads(subprocess.check_output(['docker','exec',cid,'python3','-c',code],timeout=20));out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False);wanted=['/vllm-workspace/vllm-ascend/vllm_ascend/ascend_forward_context.py','/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/routed_experts.py'];rows=[]
for path in wanted:
 q=subprocess.run(['docker','exec',cid,'cat',path],capture_output=True,timeout=20)
 if q.returncode:rows.append({'path':path,'error':q.stderr.decode()});continue
 b=q.stdout;assert len(b)<2*1024**2;name=Path(path).name;name=('spec-' if '/spec_decode/' in path else '')+name;(out/name).write_bytes(b);rows.append({'path':path,'local':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
(out/'source-report.json').write_text(json.dumps({'info':info,'files':rows},indent=2)+'\n');print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'info':info,'files':rows,'compute_contexts_created':False,'scope':'stdlib-only CPU source/config read from existingmodelcontainer, no model import or new NPU context.'})+'\nGLM53_RESULT_JSON_END')
