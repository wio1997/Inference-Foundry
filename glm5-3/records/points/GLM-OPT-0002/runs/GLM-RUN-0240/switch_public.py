from common import *
old=p/"runs/GLM-RUN-0239";proof=json.loads((old/"restored/public_service_proof.json").read_text())
retire(proof,old/"restored/public.gateway.log","retire239_public")
obsdir=old/"restored/identity_observer"
for _ in range(30):
 guard()
 if(obsdir/"terminal.json").exists():break
 time.sleep(1)
else:raise RuntimeError("old public owner observer did not terminate")
start_public(r/"restored","public240")
