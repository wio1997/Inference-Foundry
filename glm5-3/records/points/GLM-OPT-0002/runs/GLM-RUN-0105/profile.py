import sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
import json,sys,time,hashlib,urllib.request,urllib.error,re,subprocess,shlex
from pathlib import Path
from phase_runner import atomic_json,utc
from sse_observer import NativeSSEObserver
from owner_guard import start_watchdog,guard
def ownercheck():
 for x in owners.values():
  code="import pathlib,json;p=pathlib.Path('/proc/"+str(x["pid"])+"');s=(p/'stat').read_text();print(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':s[s.rfind(')')+2:].split()[19],'argv':[x.decode()for x in(p/'cmdline').read_bytes().split(bytes([0]))if x]}))";args=["python3","-c",code]
  if x["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  p=subprocess.run(args,capture_output=True,timeout=20);p.check_returncode();i=json.loads(p.stdout);assert {k:i[k]for k in ["boot_id","start_ticks"]}==x["identity"]and i["argv"]==x["argv"]
def snapshot(label):
 guard();ownercheck()
 for key,x in owners.items():
  for ep in ["health","metrics"]:
   with http.open("http://172.16.10."+x["host"]+":"+str(x["port"])+"/"+ep,timeout=10)as z:
    assert z.status==200;(r/(label+"_"+key+"."+ep)).write_bytes(z.read())
def req(name,key,body,kind,expected,min_prompt=None):
 guard();ownercheck();x=owners[key];f=r/(name+".body.json");atomic_json(f,body);raw=b"";start=time.monotonic();first=None
 row={"name":name,"owner":key,"kind":kind,"attempted_at":utc(),"request_body_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"expected_output_tokens":expected,"completed":False,"effective_public_output_credit":0};rows.append(row);atomic_json(r/"attempts.json",rows)
 request=urllib.request.Request("http://172.16.10."+x["host"]+":"+str(x["port"])+"/v1/chat/completions",data=f.read_bytes(),headers={"Content-Type":"application/json","X-Request-ID":"GLM-RUN-0105-"+name})
 try:
  with http.open(request,timeout=600)as response:
   row["http_status"]=response.status
   while True:
    block=response.read1(65536)
    if not block:break
    raw+=block
    if first is None and body.get("stream"):
     for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
      data=b"\n".join(v[5:].removeprefix(b" ")for v in frame.splitlines()if v.startswith(b"data:"))
      try:o=json.loads(data)
      except (ValueError,UnicodeDecodeError):continue
      for c in o.get("choices",[]):
       d=c.get("delta")or{}
       if any(d.get(z)for z in ["content","reasoning","reasoning_content","tool_calls"]):first=time.monotonic();first_ready.set();break
  assert row["http_status"]==200
  if body.get("stream"):
   obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True);obs.feed(raw);contract=obs.contract();assert contract["done"]and not contract["native_error"]and not contract["unknown"];usage=contract["usage"];assert contract["finish_reasons"].get("0")=="length";value=None
  else:
   value=json.loads(raw);assert not value.get("error");usage=value["usage"];assert len(value["choices"])==1 and value["choices"][0]["finish_reason"]=="length";contract={"nativeJSON":True,"finish_reason":"length"}
  assert type(usage["completion_tokens"])is int and usage["completion_tokens"]==expected and usage["total_tokens"]==usage["prompt_tokens"]+usage["completion_tokens"]
  if min_prompt is not None:assert min_prompt<usage["prompt_tokens"]<144352
  row.update(completed=True,usage=usage,contract=contract,wall_s=time.monotonic()-start,ttft_s=None if first is None else first-start,effective_public_output_credit=0 if kind=="internal_P_KV_helper"else expected)
  return value
 except urllib.error.HTTPError as e:
  raw=e.read();row.update(http_status=e.code,error="nativeHTTPerror");raise
 except BaseException as e:row.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  f=r/(name+".wire");f.write_bytes(raw);row["wire"]={"path":str(f),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()};atomic_json(r/"attempts.json",rows)
import threading,os,socket
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from sse_observer import NativeSSEObserver
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;http=urllib.request.build_opener(urllib.request.ProxyHandler({}));rows=[];owners=json.loads((r/"adopted_model_identities.json").read_text());first_ready=threading.Event()


remote_root="/data/tiankuan/wio/glm52-pd/deploy/private/profile_run105"
members=json.loads((r/"native_member_identities.json").read_text())["D1"]
workerlist=[x for x in members["owned_targets"]if x["pid"]in members["npu_worker_pids"]]
def remote(argv,input=None,timeout=60):
 guard();z=subprocess.run(["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(argv)],input=input,capture_output=True,timeout=timeout);z.check_returncode();return z.stdout
gate_code="""from pathlib import Path
import json,sys,subprocess,re
a=json.load(sys.stdin);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();out=[]
for w in a:
 p=Path('/proc/'+str(w['pid']));s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split();assert v[0]not in ['Z','X']and dict(boot_id=boot,start_ticks=v[19])==w['identity']
 e=dict(v.decode().split('=',1)for v in(p/'environ').read_bytes().split(bytes([0]))if b'='in v);assert e.get('PROFILING_MODE')=='dynamic'
 ns=next(l for l in(p/'status').read_text().splitlines()if l.startswith('NSpid:')).split()[1:];out.append(dict(host_pid=w['pid'],container_pid=int(ns[-1]),identity=w['identity'],cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()))
text=subprocess.check_output(['npu-smi','info'],text=True,timeout=60);actual={int(x)for x in re.findall(r'^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker',text,re.M)};assert actual=={x['host_pid']for x in out}
code="from pathlib import Path;import json;print(json.dumps([p.name for p in Path('/root').glob('dynamic_profiling_socket_*')]))"
socks=json.loads(subprocess.check_output(['docker','exec','glm52-single','python3','-c',code]));assert all('dynamic_profiling_socket_'+str(x['container_pid'])in socks for x in out);assert len(out)==16
print(json.dumps(dict(workers=out,sockets=socks,owned16_only=True)))
"""
gate=json.loads(remote(["python3","-c",gate_code],json.dumps(workerlist).encode()));atomic_json(r/"profile_gate.json",gate)
remote(["python3","-c","from pathlib import Path;p=Path("+repr(remote_root)+");assert not p.exists();p.mkdir()"])
snapshot("initial")
argv=["docker","exec","-i","glm52-single","/usr/local/Ascend/ascend-toolkit/latest/bin/msprof","--dynamic=on","--pid="+",".join(str(x["container_pid"])for x in gate["workers"]),"--output="+remote_root,"--runtime-api=on","--task-time=l1","--hccl=on","--aic-metrics=PipeUtilization"]
atomic_json(r/"profile_command.json",dict(at=utc(),node="167",argv=argv,model_signals=0,window_after_first_output=True))
pf=(r/"msprof.stdout").open("wb");ef=(r/"msprof.stderr").open("wb");proc=None;client=None;errors=[];control=[];stop_sent=False
def control_write(value):
 if proc is not None and proc.poll()is None:
  proc.stdin.write((value+"\n").encode());proc.stdin.flush();control.append(dict(command=value,at=utc(),monotonic=time.monotonic()));atomic_json(r/"profile_control.json",control)
def client_work():
 try:
  body=dict(model="glm-52",messages=[dict(role="user",content="List three properties of a correct inference service.")],temperature=0,seed=20260930,ignore_eos=True,max_tokens=1024,stream=True,stream_options=dict(include_usage=True),return_token_ids=True,cache_salt=r.name+"-profile-C1")
  req("profile_C1","D1",body,"public_D_profiled_decode_C1",1024);assert rows[-1]["usage"]["prompt_tokens"]==21
 except BaseException as e:errors.append(dict(type=type(e).__name__,error=str(e)))
try:
 guard();proc=subprocess.Popen(["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(argv)],stdin=subprocess.PIPE,stdout=pf,stderr=ef,start_new_session=True)
 atomic_json(r/"owned_profiler_client.json",dict(host=socket.gethostname(),local_ssh_pid=proc.pid,remote_owned_worker_namespace_pids=[x["container_pid"]for x in gate["workers"]],node="167",at=utc()))
 # msprof accepts queued stdin commands; collection begins only after real first output.
 time.sleep(2);assert proc.poll()is None
 client=threading.Thread(target=client_work);client.start()
 assert first_ready.wait(30),"no_real_first_output_before_profile"
 control_write("start");time.sleep(5);control_write("stop");stop_sent=True;control_write("quit");proc.stdin.close()
 rc=proc.wait(timeout=120);assert rc==0,("msprof_exit",rc)
 client.join(120);assert not client.is_alive()and not errors,errors
 snapshot("after")
 row=rows[0];wire=Path(row["wire"]["path"]).read_bytes();frames=[]
 for part in wire.replace(b"\r\n",b"\n").split(b"\n\n"):
  d=b"\n".join(l[5:].removeprefix(b" ")for l in part.splitlines()if l.startswith(b"data:"))
  if d and d!=b"[DONE]":frames.append(json.loads(d))
 assert sum(len(c.get("token_ids")or[])for f in frames for c in f.get("choices",[]))==1024 and any(len(f.get("prompt_token_ids")or[])==21 for f in frames)
 indexcode="""from pathlib import Path
import json,hashlib,sys
root=Path(sys.argv[1]);files=[]
for f in sorted(root.rglob('*')):
 if f.is_file():
  h=hashlib.sha256()
  with f.open('rb')as z:
   for b in iter(lambda:z.read(1048576),b''):h.update(b)
  files.append(dict(path=str(f),bytes=f.stat().st_size,sha256=h.hexdigest()))
out=dict(root=str(root),files=files,total_bytes=sum(x['bytes']for x in files),PROF_dirs=[str(x)for x in root.glob('PROF_*')if x.is_dir()])
(root/'raw_index.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
"""
 idx=json.loads(remote(["python3","-c",indexcode,remote_root],timeout=180));atomic_json(r/"profile_index.json",idx)
 captured=bool(idx["PROF_dirs"])and idx["total_bytes"]>0
 out=dict(at=utc(),functional_acceptance=True,profile_data_present=captured,profile_rank_coverage="unknown pending metadata/trace audit",effective_public_output_tokens=1024,requests=rows,control=control,profile_gate=gate,remote_raw=idx,model_operations=0,client_SDK_calls=0,native_model_SDK_init0_reused="Run104_actual_native_startup",verdict="INCONCLUSIVE",limits=["Profiled C1 decode only after first real output, no P inference; profiling overhead included in request wall, not ordinary throughput/KEEP/stablecapacity/hardwarebound","Interactive start/stop/quit requested; raw process/rank coverage, device clocks, kernel categorization and confirmed windows require independent reduction","No SDK API used by stdlib HTTP client; native resident model initialized SDK, msprof is native profiling CLI; no synthetic init/final claims"])
 atomic_json(r/"profile_summary.json",out);print(json.dumps(dict(outputs=1024,profile_data_present=captured,total_bytes=idx["total_bytes"],PROF_dirs=len(idx["PROF_dirs"]))))
finally:
 if proc is not None and proc.poll()is None:
  try:
   if not stop_sent:control_write("stop")
   control_write("quit");proc.stdin.close();proc.wait(timeout=45)
  except BaseException as e:
   atomic_json(r/"profile_shutdown_unknown.json",dict(at=utc(),error=str(e),local_owned_ssh_pid=proc.pid,warning="Do not treat collection as stopped; reconcile before nextGPUwork"))
 if client is not None:client.join(120)
 pf.close();ef.close()
