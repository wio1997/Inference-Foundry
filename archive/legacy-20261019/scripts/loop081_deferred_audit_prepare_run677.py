"""Prepare from restored source; no source installation."""
import hashlib,json
from pathlib import Path
import loop081_deferred_audit_patch_run677 as edit
root=Path('/data/wio/Inference_Foundry');out=root/'evidence/20260929_loop081_bound/run677'
sha=lambda b:hashlib.sha256(b).hexdigest()
base=json.loads((root/'evidence/20260929_loop081_bound/run675/candidate_manifest.json').read_text())
for name,expected in base['helpers'].items():assert sha(Path(name).read_bytes())==expected
pin={'sources':{},'helpers':{}};prepared={}
for key,row in base['sources'].items():
 p=Path(row['path']);raw=p.read_bytes();assert sha(raw)==row['before_sha256']
 b=(root/f'evidence/20260929_loop081_bound/run675/candidate/{key}.py').read_bytes();assert sha(b)==row['candidate_sha256'];s=b.decode()
 if key=='runner':
  s=edit.one(s,'                    "dspark_slot_refresh_audit": _extreme_runtime.proposer.slot_refresh_audit,',
   '''                    "dspark_slot_refresh_audit": _extreme_runtime.proposer.slot_refresh_audit,
                    "slot_audit_mode": dict(_extreme_runtime.proposer._slot_audit_mode),
                    "slot_audit_pending": len(_extreme_runtime.proposer._pending_slot_refresh_audit),''')
 compile(s,str(p),'exec');prepared[key]=s;pin['sources'][key]={'path':str(p),'before_sha256':sha(raw),'candidate_sha256':sha(s.encode())}
p=root/'bootstrap/vllm_dspark_handoff.py';raw=p.read_bytes();assert sha(raw)=='fe7039361894acd70666f72c3089efd635404f1fe9cbc685db21272d946a317e'
s=edit.patch(raw.decode());compile(s,str(p),'exec');prepared['handoff']=s;pin['sources']['handoff']={'path':str(p),'before_sha256':sha(raw),'candidate_sha256':sha(s.encode())}
for p in (Path(edit.__file__),Path(__file__).resolve(),root/'serving/cohort_mode.py',root/'scripts/loop081_runtime_observer_install_run638.py'):
 pin['helpers'][str(p)]=sha(p.read_bytes())
cand=out/'candidate';cand.mkdir(parents=True,exist_ok=False)
for key,s in prepared.items():(cand/f'{key}.py').write_text(s)
(out/'candidate_manifest.json').write_text(json.dumps(pin,indent=2)+'\n')
print(json.dumps({'prepared':True,'files':3,'no_install':True}))
