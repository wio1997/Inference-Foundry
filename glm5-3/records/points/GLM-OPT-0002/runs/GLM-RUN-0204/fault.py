from common import *
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
fresh(roots["node1"],members["node1"],"before_fault_D1");v=fresh(roots["node0"],members["node0"],"before_fault_D0")
for node,port in[("166",9081),("167",9900)]:native_idle(node,port,"before_fault")
assert request("/healthcheck")["request_num"]==0
payload=dict(owner=roots["node0"],members=v["owned_targets"],preflight=True)
run("166",["python3","-c",(r/"stop_prior_cohort.py").read_text()],"D0_stop_preflight",input=json.dumps(payload).encode(),timeout=180);payload["preflight"]=False
run("166",["python3","-c",(r/"stop_prior_cohort.py").read_text()],"D0_stop",input=json.dumps(payload).encode(),timeout=180)
obsdir=p/"runs/GLM-RUN-0202/restored/identity_observer";deadline=time.monotonic()+90
while True:
 guard();snap=request("/control/replicas");a={x["id"]:x for x in snap["replicas"]};row=json.loads((obsdir/"latest.json").read_text())
 if a["D0"]["group_faulted"]and next(x for x in row["groups"]if x["id"]=="local-202-166")["status"]=="fault":break
 assert time.monotonic()<deadline;time.sleep(1)
assert not a["D1"]["group_faulted"]and all(not x["active_requests"]for x in snap["replicas"])
fault=json.loads((state/"response_owners.json.fault").read_text());assert fault["groups"]["local-202-166"]["faulted"]and not fault["groups"]["local-200-167"]["faulted"]
atomic_json(r/"physical_fault_ack.json",dict(at=utc(),observer=row,placement=snap,journal=fault,confirmed_native_D0_identity_loss=True,D1_retained=True,source="automaticHOSTobserver",native_STOP_scope="166exactowned202_domainonly"))
client("survivor");fresh(roots["node1"],members["node1"],"after_fault_D1");native_idle("167",9900,"after_fault")
