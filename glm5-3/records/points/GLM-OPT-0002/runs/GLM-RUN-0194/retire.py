from common import *
old=json.loads((r/"old_public_service_proof.json").read_text());retire(old,p/"runs/GLM-RUN-0190/restored/public.gateway.log","retire190_public")
obsdir=p/"runs/GLM-RUN-0190/restored/identity_observer"
for _ in range(30):
 guard()
 if(obsdir/"terminal.json").exists():break
 time.sleep(1)
else:raise RuntimeError("oldobserver didnot end")
atomic_json(r/"reused193_retirement.json",dict(at=utc(),partial193_fault_ack=ref(p/"runs/GLM-RUN-0193/physical_fault_ack.json"),retirement_complete=True,replayed_signals=0))
