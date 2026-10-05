from common import *
old=json.loads((r/"old_public_service_proof.json").read_text());retire(old,p/"runs/GLM-RUN-0219/restored/public.gateway.log","retire219_public")
obsdir=p/"runs/GLM-RUN-0219/restored/identity_observer"
for _ in range(30):
 guard()
 if(obsdir/"terminal.json").exists():break
 time.sleep(1)
else:raise RuntimeError("old HOSTobserver didnot end with exactpublicowner")
assert json.loads((state/"response_owners.json.fault").read_text())["groups"]["local-219-166"]["faulted"]
folder=r/"restored";start_public(folder,"public221");atomic_json(r/"diagnostic_enable.json",dict(enabled=True,diagnostic_only=True))
