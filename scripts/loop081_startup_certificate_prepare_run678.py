"""Prepare Run678 from restored sources; never install on import."""
import hashlib,json
from pathlib import Path
import loop081_startup_certificate_patch_run678 as edit
root=Path('/data/wio/Inference_Foundry');out=root/'evidence/20260929_loop081_bound/run678'
sha=lambda b:hashlib.sha256(b).hexdigest()
base=json.loads((root/'evidence/20260929_loop081_bound/run677/candidate_manifest.json').read_text())
for name,expected in base['helpers'].items():assert sha(Path(name).read_bytes())==expected
pin={'sources':{},'helpers':{}};prepared={}
for key,row in base['sources'].items():
 p=Path(row['path']);raw=p.read_bytes();assert sha(raw)==row['before_sha256']
 b=(root/f'evidence/20260929_loop081_bound/run677/candidate/{key}.py').read_bytes();assert sha(b)==row['candidate_sha256'];s=b.decode()
 if key=='runner':s=edit.runner(s)
 elif key=='handoff':s=edit.handoff(s)
 compile(s,str(p),'exec');prepared[key]=s;pin['sources'][key]={'path':str(p),'before_sha256':sha(raw),'candidate_sha256':sha(s.encode())}
p=root/'runtime/fixed_serving.py';raw=p.read_bytes()
import subprocess
assert raw==subprocess.check_output(['git','show','51d0b34b:runtime/fixed_serving.py'],cwd=root)
s=edit.serving(raw.decode());compile(s,str(p),'exec');prepared['serving']=s
pin['sources']['serving']={'path':str(p),'before_sha256':sha(raw),'candidate_sha256':sha(s.encode())}
helpers=[Path(edit.__file__),Path(__file__).resolve(),root/'serving/cohort_mode.py',root/'scripts/loop081_runtime_observer_install_run638.py',root/'runtime/startup_slot_certificate.py',root/'runtime/fixed_decode.py',root/'runtime/extreme_decode.py',Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py')]
for p in helpers:pin['helpers'][str(p)]=sha(p.read_bytes())
cand=out/'candidate';cand.mkdir(parents=True,exist_ok=False)
for key,s in prepared.items():(cand/f'{key}.py').write_text(s)
(out/'candidate_manifest.json').write_text(json.dumps(pin,indent=2)+'\n')
print(json.dumps({'prepared':True,'files':len(prepared),'no_install':True}))
