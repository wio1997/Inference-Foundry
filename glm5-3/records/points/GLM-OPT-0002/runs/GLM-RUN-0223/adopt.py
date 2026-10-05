from common import *
folder=r/"restored";roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text())
for key,root in roots.items():fresh(root,members[key],"adopt_"+key)
proof=json.loads((folder/"public_service_proof.json").read_text());assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
config,_=checked_config(folder/"service_config.json");old,_=checked_config(p/"runs/GLM-RUN-0222/restored/service_config.json");assert config==old
assert not(p/"runs/GLM-RUN-0222/diagnostic_enable.json").exists()
atomic_json(r/"adoption_summary.json",dict(at=utc(),native32_same_roots=True,same_native_epochs=True,same_public222=True,draftNONE_and_targetFULL_proof=ref(folder/"draft_eager_dispatch_proof.json"),models_started=0,models_signalled=0,public_restarts=0))
