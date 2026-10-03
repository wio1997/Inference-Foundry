"""Host-side observation of recorded native process identities.

Confirmed identity loss quarantines only its recorded epoch through the gateway.
SSH or unreadable-proc failures are unknown and never masquerade as identity loss.
No native process, model, lease, state payload, or inference is changed here.
"""
import argparse,json,subprocess,shlex,time,urllib.request,urllib.error
from pathlib import Path
from native_engines_service_config import checked_config
from phase_runner import atomic_json,utc,same_process

HOST_CHECK = r"""
import json,sys
from pathlib import Path
a=json.load(sys.stdin);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def process(expected,argv=None):
 pid=expected['pid'];identity=expected['identity'];p=Path('/proc')/str(pid)
 try:raw=(p/'stat').read_text()
 except FileNotFoundError:return dict(status='fault',cause='pid_missing',pid=pid)
 vals=raw[raw.rfind(')')+2:].split()
 observed=dict(boot_id=boot,start_ticks=vals[19]);status=vals[0]
 if status in ['Z','X']:return dict(status='fault',cause='terminal_process',pid=pid,observed=observed,state=status)
 if observed!=identity:return dict(status='fault',cause='identity_changed',pid=pid,observed=observed)
 if argv is not None and [x.decode()for x in(p/'cmdline').read_bytes().split(bytes([0]))if x]!=argv:
  return dict(status='fault',cause='argv_changed',pid=pid,observed=observed)
 return dict(status='healthy',pid=pid,observed=observed)
try:
 root=process(a['root'],a['root']['argv']);rows=[root]
 if root['status']=='healthy':
  for worker in a['workers']:
   row=process(worker);rows.append(row)
   if row['status']!='healthy':continue
   pid=worker['pid'];seen=set()
   while pid!=a['root']['pid'] and pid>1 and pid not in seen:
    seen.add(pid);raw=(Path('/proc')/str(pid)/'stat').read_text();pid=int(raw[raw.rfind(')')+2:].split()[1])
   if pid!=a['root']['pid']:row.update(status='fault',cause='native_worker_ancestry_changed')
 print(json.dumps(dict(status='fault'if any(x['status']=='fault'for x in rows)else'healthy',rows=rows)))
except (OSError,ValueError,IndexError,KeyError)as error:
 print(json.dumps(dict(status='unknown',error_type=type(error).__name__,error=str(error))))
"""

def host_probe(host,root,workers):
    args=["python3","-c",HOST_CHECK]
    if host=="167":args=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=5","root@172.16.10.167",shlex.join(args)]
    elif host!="166":raise ValueError("unrecognized task host")
    try:
        result=subprocess.run(args,input=json.dumps(dict(root=root,workers=workers)).encode(),capture_output=True,timeout=10)
        if result.returncode:
            return dict(status="unknown",error="host probe exit",returncode=result.returncode,stderr=result.stderr.decode(errors="replace")[-512:])
        row=json.loads(result.stdout)
        if not isinstance(row,dict)or row.get("status")not in["healthy","fault","unknown"]:
            raise ValueError("unknown host observation")
        return row
    except (OSError,subprocess.TimeoutExpired,ValueError)as error:
        return dict(status="unknown",error_type=type(error).__name__,error=str(error))

def observe(config_path,probe=host_probe):
    config,_=checked_config(config_path)
    material=json.loads(Path(config["resident_evidence"]["native_engines_resident.json"]["path"]).read_text())
    domains={x["id"]:x for x in config["native_domains"]};groups=[]
    for engine in material["engines"]:
        domain=domains[engine["id"]];hosts=[]
        for key,root in engine["roots"].items():
            member=engine["members"][key]
            targets={x["pid"]:x for x in member["owned_targets"]}
            workers=[targets[pid]for pid in member["npu_worker_pids"]]
            row=probe(root["host"],root,workers)
            hosts.append(dict(host=root["host"],root_pid=root["pid"],expected_native_workers=len(workers),observation=row))
        statuses=[x["observation"]["status"]for x in hosts]
        status="fault"if"fault"in statuses else"unknown"if"unknown"in statuses else"healthy"
        groups.append(dict(id=domain["id"],epoch=domain["epoch"],members=domain["members"],status=status,hosts=hosts))
    return dict(at=utc(),groups=groups,limits=["Live process identity does not certify device health, inference correctness or detect every GPU hang","Unavailable transport or proc observation is unknown, never a confirmed native identity loss","An epoch fault is sticky; only a newly reconciled physical owner epoch permits replacement"])

def owner_alive(owner):
    identity=owner["identity"]
    if not same_process(identity):return False
    actual=[x.decode()for x in Path("/proc") .joinpath(str(identity["pid"]),"cmdline").read_bytes().split(bytes([0]))if x]
    return actual==owner["argv"]

def apply_observation(observation,public_owner,base_url,post):
    """Epoch matching at the gateway makes delayed observations idempotent."""
    if not owner_alive(public_owner):raise RuntimeError("recorded public owner no longer alive")
    actions=[]
    for group in observation["groups"]:
        if group["status"]!="fault":continue
        body=dict(epoch=group["epoch"],reason="host observer confirmed recorded native process identity loss")
        if not owner_alive(public_owner):raise RuntimeError("recorded public owner no longer alive")
        actions.append(dict(group=group["id"],epoch=group["epoch"],response=post(base_url+"/control/native-groups/"+group["id"]+"/fault",body)))
    return actions

def http_post(url,body):
    request=urllib.request.Request(url,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"},method="POST")
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request,timeout=10)as response:raw=response.read();status=response.status
    except urllib.error.HTTPError as error:raw=error.read();status=error.code
    result=dict(status=status,body=json.loads(raw))
    if status!=200 or result["body"].get("faulted")is not True:raise RuntimeError("native quarantine ACK unavailable: "+str(status))
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--public-owner",type=Path)
    parser.add_argument("--base-url",default="http://127.0.0.1:8000")
    parser.add_argument("--interval-s",type=float,default=5)
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    if args.interval_s<=0:raise ValueError("positive observation interval required")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    owner=json.loads(args.public_owner.read_text())if args.public_owner else None
    while True:
        if owner is not None and not owner_alive(owner):
            atomic_json(args.output_dir/"terminal.json",dict(at=utc(),reason="public owner ended",native_signals=0));return
        row=observe(args.config);row["actions"]=[]
        if owner is not None:
            try:row["actions"]=apply_observation(row,owner,args.base_url,http_post)
            except (OSError,ValueError,RuntimeError)as error:row["apply_error"]=dict(type=type(error).__name__,error=str(error))
        atomic_json(args.output_dir/"latest.json",row)
        with(args.output_dir/"observations.jsonl").open("a")as stream:stream.write(json.dumps(row)+"\n")
        if args.once:return
        time.sleep(args.interval_s)

if __name__=="__main__":main()
