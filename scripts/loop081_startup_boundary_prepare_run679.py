"""Prepare immutable diagnostic candidates from restored Run678 base sources."""
import hashlib,json
from pathlib import Path
import loop081_startup_boundary_patch_run679 as edit
import loop081_startup_boundary_core_patch_run679 as core_edit
root=Path('/data/wio/Inference_Foundry');out=root/'evidence/20260929_loop081_bound/run679'
sha=lambda b:hashlib.sha256(b).hexdigest()
base=json.loads((root/'evidence/20260929_loop081_bound/run678/candidate_manifest.json').read_text())
for name,expected in base['helpers'].items():assert sha(Path(name).read_bytes())==expected,name
pin={'sources':{},'helpers':{}};prepared={}
for key,row in base['sources'].items():
 p=Path(row['path']);raw=p.read_bytes();assert sha(raw)==row['before_sha256'],key
 b=(root/f'evidence/20260929_loop081_bound/run678/candidate/{key}.py').read_bytes();assert sha(b)==row['candidate_sha256'];s=b.decode()
 if key=='runner':s=edit.runner(s)
 elif key=='handoff':s=edit.handoff(s)
 compile(s,str(p),'exec');prepared[key]=s;pin['sources'][key]={'path':str(p),'before_sha256':sha(raw),'candidate_sha256':sha(s.encode())}
p=Path('/data/wio/vllm_ascend_26/framework/vllm/vllm/v1/engine/core.py');raw=p.read_bytes()
assert sha(raw)=='3ae1381a6af841e21058c825702382dc66faae45c950ac5acb8495d2d3d05aad'
s=core_edit.core(raw.decode());compile(s,str(p),'exec');prepared['core']=s
pin['sources']['core']={'path':str(p),'before_sha256':sha(raw),'candidate_sha256':sha(s.encode())}
helpers=list(base['helpers'])+[str(Path(edit.__file__)),str(Path(core_edit.__file__)),str(Path(__file__).resolve()),str(root/'scripts/loop081_ordinary_boundary_patch_run676.py'),str(root/'diagnostics/startup_boundary/observer.py')]
for name in helpers:pin['helpers'][name]=sha(Path(name).read_bytes())
cand=out/'candidate';cand.mkdir(parents=True,exist_ok=False)
for key,s in prepared.items():(cand/f'{key}.py').write_text(s)
(out/'candidate_manifest.json').write_text(json.dumps(pin,indent=2)+'\n')
print(json.dumps({'prepared':True,'files':len(prepared),'no_install':True}))
