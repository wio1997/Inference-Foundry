from common import *
old=json.loads((r/"old_public_service_proof.json").read_text());retire(old,p/"runs/GLM-RUN-0211/restored/public.gateway.log","retire211_public")
obsdir=p/"runs/GLM-RUN-0211/restored/identity_observer"
for _ in range(30):
 guard()
 if(obsdir/"terminal.json").exists():break
 time.sleep(1)
else:raise RuntimeError("old HOSTobserver didnot end with exactpublicowner")
assert json.loads((state/"response_owners.json.fault").read_text())["groups"]["local-204-166"]["faulted"]
folder=r/"restored";start_public(folder,"public214");client("final")
roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"final_"+key);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"final")
atomic_json(r/"functional_final.json",dict(at=utc(),valid=True,actual_native32=True,D1_same_epoch=True,D0_new_epoch=True,public=ref(folder/"public_service_proof.json"),SDKclients0=True,PDhelpers=0,STORE_replication=False,request_replay=False,capacity_claim=False))
