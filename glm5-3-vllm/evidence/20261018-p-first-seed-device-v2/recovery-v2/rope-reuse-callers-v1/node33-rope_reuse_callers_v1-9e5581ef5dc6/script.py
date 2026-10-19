"""Inventory exact-image RoPE callers without importing model/device modules."""
from pathlib import Path
import hashlib, json, socket, subprocess
assert socket.gethostname() == 'B5-5U-Atlas900RCKA3-33'
cid = 'bfac6522d752e0595df01f537e2f74bbe4c4d00cf9334668283e6885b91e910d'
d = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
assert d['State']['Running'] and d['Name'] == '/glm53-early-publication-33-live2-20261019'
assert d['Image'] == 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
probe = Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-publication-live2-20261019/node33/early-sample-publication-probe')
assert not (probe/'enable').exists() and (probe/'disabled').exists()
code = """from pathlib import Path
import json
root=Path('/vllm-workspace')
rows=[]
for package in ['vllm-ascend/vllm_ascend','vllm/vllm']:
 for p in (root/package).rglob('*.py'):
  text=p.read_text(errors='replace')
  if 'get_cos_and_sin_mla' in text:
   rows.append(dict(path=str(p.relative_to(root)),lines=[dict(line=i+1,text=s.strip()) for i,s in enumerate(text.splitlines()) if 'get_cos_and_sin_mla' in s]))
print(json.dumps(rows))
"""
rows = json.loads(subprocess.check_output(['docker', 'exec', cid, 'python3', '-c', code], timeout=40))
out = Path(__file__).resolve().parent/'artifacts'
out.mkdir(exist_ok=False)
for row in rows:
    data = subprocess.check_output(['docker', 'exec', cid, 'cat', '/vllm-workspace/'+row['path']], timeout=20)
    target = out/row['path']; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(dict(completed=True,CID=cid,callers=rows,scope='Read-only exact installed source. Inventory only, no model imports, flags, device work or performance claim.'))+'\nGLM53_RESULT_JSON_END')
