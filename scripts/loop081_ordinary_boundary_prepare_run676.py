"""Prepare next diagnostic from original files only after Run675 restores them."""
import hashlib,json
from pathlib import Path
import loop081_ordinary_boundary_patch_run676 as patch
root=Path('/data/wio/Inference_Foundry');out=root/'evidence/20260929_loop081_bound/run676'
sha=lambda b:hashlib.sha256(b).hexdigest()
base=json.loads((root/'evidence/20260929_loop081_bound/run675/candidate_manifest.json').read_text())
# Ensure historical selector/arithmetic source being reused is the reviewed candidate.
for name,expected in base['helpers'].items():assert sha(Path(name).read_bytes())==expected,name
pin={'sources':{},'helpers':{}}
prepared={}
for key,row in base['sources'].items():
 path=Path(row['path']);raw=path.read_bytes();assert sha(raw)==row['before_sha256'],key
 candidate=(root/f'evidence/20260929_loop081_bound/run675/candidate/{key}.py').read_bytes();assert sha(candidate)==row['candidate_sha256']
 new=patch.runner(candidate.decode()) if key=='runner' else candidate.decode()
 compile(new,str(path),'exec');prepared[key]=new
 pin['sources'][key]={'path':str(path),'before_sha256':sha(raw),'candidate_sha256':sha(new.encode())}
path=Path('/data/wio/vllm_ascend_26/framework/vllm/vllm/v1/engine/core.py');raw=path.read_bytes()
assert sha(raw)=='3ae1381a6af841e21058c825702382dc66faae45c950ac5acb8495d2d3d05aad'
new=patch.core(raw.decode());compile(new,str(path),'exec');prepared['core']=new
pin['sources']['core']={'path':str(path),'before_sha256':sha(raw),'candidate_sha256':sha(new.encode())}
helpers=[root/'diagnostics/ordinary_boundary/observer.py',Path(patch.__file__),Path(__file__).resolve(),root/'serving/cohort_mode.py',root/'scripts/loop081_runtime_observer_install_run638.py']
for p in helpers:pin['helpers'][str(p)]=sha(p.read_bytes())
cand=out/'candidate';cand.mkdir(parents=True,exist_ok=False)
for key,new in prepared.items():(cand/(key+'.py')).write_text(new)
(out/'candidate_manifest.json').write_text(json.dumps(pin,indent=2)+'\n')
print(json.dumps({'prepared':True,'files':len(prepared),'no_install':True}))
