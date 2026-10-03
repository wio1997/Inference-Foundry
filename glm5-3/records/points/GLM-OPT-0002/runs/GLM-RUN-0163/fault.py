from common import *
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
fresh(roots["node0"],members["node0"],"before_fault_D0");v=fresh(roots["node1"],members["node1"],"before_fault_D1")
native_idle("166",9081,"before_fault");native_idle("167",9900,"before_fault");assert request("/healthcheck")["request_num"]==0
payload=dict(owner=roots["node1"],members=v["owned_targets"],preflight=True)
run("167",["python3","-c",(r/"stop_prior_cohort.py").read_text()],"D1_stop_preflight",input=json.dumps(payload).encode(),timeout=180)
payload["preflight"]=False
run("167",["python3","-c",(r/"stop_prior_cohort.py").read_text()],"D1_stop",input=json.dumps(payload).encode(),timeout=180)
obsdir=p/"runs/GLM-RUN-0156/restored/identity_observer"
for _ in range(90):
 guard();row=json.loads((obsdir/"latest.json").read_text());actions=row.get("actions",[])
 if any(x["group"]=="local-167"and x["response"]["status"]==200for x in actions):break
 time.sleep(1)
else:raise RuntimeError("physicalD1 loss not automatically quarantined")
assert next(x for x in row["groups"]if x["id"]=="local-166")["status"]=="healthy"and next(x for x in row["groups"]if x["id"]=="local-167")["status"]=="fault"
snap=request("/control/replicas");a={x["id"]:x for x in snap["replicas"]};assert a["D1"]["group_faulted"]and not a["D0"]["group_faulted"]and all(not x["active_requests"]for x in snap["replicas"])
fault=json.loads((state/"response_owners.json.fault").read_text());assert fault["groups"]["local-167"]["faulted"]and not fault["groups"]["local-166"]["faulted"]
atomic_json(r/"physical_fault_ack.json",dict(at=utc(),observer=row,placement=snap,journal=fault,confirmed_native_D1_identity_loss=True,D0_retained=True,source="automaticHOSTobserver",native_STOP_scope="167owned156domainonly"))
client("survivor")
fresh(roots["node0"],members["node0"],"after_fault_D0");native_idle("166",9081,"after_fault")
