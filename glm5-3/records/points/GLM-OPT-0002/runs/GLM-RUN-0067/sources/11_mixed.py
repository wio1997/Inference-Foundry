from pathlib import Path
import json,sys,time,hashlib,urllib.request,urllib.error,threading,concurrent.futures,subprocess,shlex,re,statistics
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
from sse_observer import NativeSSEObserver
start_watchdog();r=Path(__file__).parent;owners=json.loads((r/"adopted_model_identities.json").read_text());proof=json.loads((r/"native_member_identities.json").read_text());lock=threading.Lock();rows={};http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def ownercheck():
 for key,o in owners.items():
  targets=[x for x in proof[key]["owned_targets"]if x["pid"]in proof[key]["npu_worker_pids"]];code="""import json,pathlib,sys
a=json.load(sys.stdin);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for x in a['targets']:
 p=pathlib.Path('/proc/'+str(x['pid']));s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split();assert v[0]not in['Z','X']and dict(boot_id=boot,start_ticks=v[19])==x['identity']
o=a['owner'];assert [v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv'];print(json.dumps(dict(checked=len(a['targets']),all_same=True)))
"""
  args=["python3","-c",code]
  if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,input=json.dumps(dict(owner=o,targets=targets)).encode(),capture_output=True,timeout=30);z.check_returncode()
def snap(label):
 guard();ownercheck()
 for key,o in owners.items():
  with http.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as z:b=z.read();assert z.status==200;(r/(label+"_"+key+".metrics")).write_bytes(b)
def update():
 with lock:atomic_json(r/"attempts.json",list(rows.values()))
canonical=json.loads((r/"canonical_81932.body.json").read_text());cases=[]
for ident,key,offset,output,long in [("long_D1","D1",0.,256,True),("short0_D0","D0",.1,64,False),("short1_D0","D0",.3,128,False),("short2_D0","D0",.5,256,False),("short_D1","D1",.6,64,False)]:
 b=dict(canonical)if long else dict(model="glm-52",messages=[dict(role="user",content="List three properties of a correct inference service.")],temperature=0,seed=20260930,ignore_eos=True)
 b.update(max_tokens=output,stream=True,stream_options=dict(include_usage=True),return_token_ids=True,cache_salt=r.name+"-"+ident);b.pop("kv_transfer_params",None)
 f=r/(ident+".body.json");atomic_json(f,b);cases.append(dict(id=ident,key=key,arrival_s=offset,body=f.name,expected_outputs=output,input_kind="81932canonical"if long else"short"))
atomic_json(r/"arrival_plan.json",dict(kind="diagnostic",cases=cases,contract="Five new controlled open-arrival directnative requests; longprefillD1 vsD0 decode and D1localqueue, no router/capacity/KEEP"))
snap("before");origin=time.monotonic()
def execute(c):
 while time.monotonic()<origin+c["arrival_s"]:time.sleep(max(0,min(.01,origin+c["arrival_s"]-time.monotonic())))
 guard();key=c["key"];o=owners[key];body=(r/c["body"]).read_bytes();start=time.monotonic();row=dict(id=c["id"],owner=key,scheduled_arrival_s=c["arrival_s"],actual_dispatch_s=start-origin,attempted_at=utc(),completed=False,effective_public_output_credit=0,body_sha256=hashlib.sha256(body).hexdigest());
 with lock:rows[c["id"]]=row
 update();raw=b"";pending=b"";first=None;ids=[];points=[];obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True)
 try:
  req=urllib.request.Request("http://172.16.10."+o["host"]+":"+str(o["port"])+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json","X-Request-ID":r.name+"-"+c["id"]})
  with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=300)as response:
   
   with lock:row["http_status"]=response.status
   assert response.status==200
   while True:
    block=response.read1(65536)
    if not block:break
    now=time.monotonic();raw+=block;obs.feed(block);pending+=block
    while b"\n\n"in pending:
     frame,pending=pending.split(b"\n\n",1);d=b"\n".join(l[5:].removeprefix(b" ")for l in frame.splitlines()if l.startswith(b"data:"))
     if not d or d==b"[DONE]":continue
     x=json.loads(d);assert not x.get("error")
     for ch in x.get("choices",[]):
      delta=ch.get("delta")or{}
      if first is None and any(delta.get(k)for k in["content","reasoning","reasoning_content","tool_calls"]):first=now
      ts=ch.get("token_ids")or[]
      if ts:assert all(type(v)is int for v in ts);ids.extend(ts);points.append(dict(elapsed_s=now-start,dispatch_origin_s=now-origin,committed_token_count=len(ids),native_chunk_commits=len(ts)))
  con=obs.contract();assert con["done"]and not con["native_error"]and not con["unknown"]and con["finish_reasons"]=={"0":"length"}
  u=con["usage"];assert u["completion_tokens"]==c["expected_outputs"]and len(ids)==c["expected_outputs"]and u["total_tokens"]==u["prompt_tokens"]+u["completion_tokens"]
  if c["input_kind"]=="81932canonical":assert u["prompt_tokens"]==81932
  intervals=[dict(interval_s=points[i]["elapsed_s"]-points[i-1]["elapsed_s"],native_chunk_commits=points[i]["native_chunk_commits"])for i in range(1,len(points))]
  with lock:row.update(completed=True,usage=u,ttft_s=None if first is None else first-start,wall_s=time.monotonic()-start,effective_public_output_credit=u["completion_tokens"],committed_token_ids=ids,native_token_chunk_points=points,native_chunk_intervals=intervals,max_native_chunk_gap_s=max([v["interval_s"]for v in intervals]or[0]),contract=con)
 except BaseException as e:
  with lock:row.update(error_type=type(e).__name__,error=str(e))
  raise
 finally:
  f=r/(c["id"]+".wire");f.write_bytes(raw)
  with lock:row["wire"]=dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  update()
with concurrent.futures.ThreadPoolExecutor(max_workers=5)as pool:
 tasks=[pool.submit(execute,c)for c in cases]
 for f in tasks:f.result()
snap("after");assert len(rows)==5 and all(x["completed"]for x in rows.values())
out=dict(at=utc(),measurement_valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",actual_new_requests=5,effective_public_output_tokens=sum(x["effective_public_output_credit"]for x in rows.values()),elapsed_s=time.monotonic()-origin,requests=list(rows.values()),limits=["Finite explicitnativeendpoint open-arrival interference diagnostic; no publicgateway/allAPI/state/capacity/KEEP","Native token-chunk arrival timing includes transport/MTPchunking; chunkgap not pertoken GPU time or formalTPOTpercentile","One longprefillD1 plus3D0short and1D1short; resource/workload differssequential65, noisolatedbatch/kernel gain","New D67 epochs with native CPU issuebudget4096, GPUallocation16384/12.8GiB/seq8/HCCL4096/K5 unchanged from65; deploy rebuild authorized/exactD65cleanup; mixed phase no starts/signals/oldqueue"])
atomic_json(r/"mixed_summary.json",out);print(json.dumps(out))
