from common import *
proof=json.loads((p/"jobs/AUDIT-RUN148-20261003T0425Z/public_service_proof.json").read_text())
retire(proof,p/"runs/GLM-RUN-0147/public.gateway.log","retire147")
start_public(r,"public149")
client("before")
