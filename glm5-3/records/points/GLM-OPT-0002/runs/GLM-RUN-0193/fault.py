from common import *
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
assert request("/healthcheck")["request_num"]==0
for k,o in roots.items():fresh(o,members[k],"before_fault_"+k);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before_fault")
payloads={}
for k,o in roots.items():
 v=fresh(o,members[k],"stop_fresh_"+k);payloads[k]=dict(owner=o,members=v["owned_targets"],preflight=True)
 run(o["host"],["python3","-c",(r/"stop_prior_cohort.py").read_text()],"stop_preflight_"+k,input=json.dumps(payloads[k]).encode(),timeout=180)
for k,o in roots.items():
 payloads[k]["preflight"]=False;run(o["host"],["python3","-c",(r/"stop_prior_cohort.py").read_text()],"stop_"+k,input=json.dumps(payloads[k]).encode(),timeout=180)
obsdir=p/"runs/GLM-RUN-0190/restored/identity_observer";deadline=time.monotonic()+90
while True:
 guard();snap=request("/control/replicas");row=json.loads((obsdir/"latest.json").read_text())
 if all(x["group_faulted"]for x in snap["replicas"])and all(x["status"]=="fault"for x in row["groups"]):break
 assert time.monotonic()<deadline;time.sleep(1)
assert all(not x["active_requests"]for x in snap["replicas"])
fault=json.loads((state/"response_owners.json.fault").read_text());assert all(fault["groups"][k]["faulted"]for k in["local-166","local-167"])
atomic_json(r/"physical_fault_ack.json",dict(at=utc(),observer=row,placement=snap,journal=fault,both_exact_old_native_domains_retired=True,STORE_replication=False))
client("retired")
old=json.loads((r/"old_public_service_proof.json").read_text());retire(old,p/"runs/GLM-RUN-0190/restored/public.gateway.log","retire190_public")
for _ in range(30):
 guard()
 if(obsdir/"terminal.json").exists():break
 time.sleep(1)
else:raise RuntimeError("old HOSTobserver didnot end")
