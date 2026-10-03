from common import *
old=json.loads((r/"prepared_public_service.json").read_text())
retire(old,p/"runs/GLM-RUN-0175/restored/public.gateway.log","retire175_public")
obsdir=p/"runs/GLM-RUN-0175/restored/identity_observer"
for _ in range(30):
 guard()
 if(obsdir/"terminal.json").exists():break
 time.sleep(1)
else:raise RuntimeError("oldHOSTobserver didnot terminate")
start_public(r,"public179_V12")
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"switch_"+key);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"switch")
atomic_json(r/"switched_state.json",dict(at=utc(),V12=True,only_owned_public_replaced=True,native32_unchanged=True,state125_preserved=True,token_cache_preload=False))
