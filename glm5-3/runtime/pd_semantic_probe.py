"""Diagnostic PD/local semantic control, executed only under a task controller."""
import json,time,hashlib,re,subprocess,shlex,urllib.request,urllib.error
from pathlib import Path
from phase_runner import atomic_json,utc

COUNTERS=("request_success_total","generation_tokens_total","prefix_cache_queries_total","prefix_cache_hits_total","external_prefix_cache_queries_total","external_prefix_cache_hits_total")
def values(raw,name):
 return [float(m) for m in re.findall(r"^vllm:"+re.escape(name)+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",raw,re.M)]
def counter(raw,name):
 v=values(raw,name)
 return sum(v) if v else None
def delta(before,after,name):
 b=counter(before,name);a=counter(after,name)
 return None if b is None or a is None else a-b
def ref(path):
 b=Path(path).read_bytes()
 return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def execute(run,guard,start_watchdog):
 """No deployment, cleanup, cancellation, retry, or controller launch here."""
 r=Path(run);start_watchdog();guard()
 owners=json.loads((r/"adopted_model_identities.json").read_text());rows=[];pairs=[];http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 def ownercheck():
  guard()
  for x in owners.values():
   code="import pathlib,json;p=pathlib.Path('/proc/"+str(x["pid"])+"');s=(p/'stat').read_text();print(json.dumps(dict(boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=s[s.rfind(')')+2:].split()[19],argv=[v.decode()for v in(p/'cmdline').read_bytes().split(bytes([0]))if v])))"
   argv=["python3","-c",code]
   if x["host"]=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(argv)]
   obj=json.loads(subprocess.check_output(argv,timeout=20))
   assert {k:obj[k]for k in["boot_id","start_ticks"]}==x["identity"]and obj["argv"]==x["argv"],"native API owner changed"
 def url(key,ep):
  x=owners[key];return "http://172.16.10."+x["host"]+":"+str(x["port"])+"/"+ep
 def snapshot(label):
  ownercheck();out={}
  for key in owners:
   for ep in["health","metrics"]:
    guard()
    with http.open(url(key,ep),timeout=10)as z:
     assert z.status==200;data=z.read()
    f=r/(label+"_"+key+"."+ep);f.write_bytes(data)
    if ep=="metrics":
     text=data.decode()
     for name in["num_requests_running","num_requests_waiting"]:
      v=values(text,name);assert v and all(x==0 for x in v),("native not idle",key,name,v)
     out[key]=text
  return out
 def settle(label,key,before):
  end=time.monotonic()+45;idx=0
  while True:
   # Failed native engines or another arrival fail closed through snapshot.
   snap=snapshot(label+"_"+str(idx));n=delta(before,snap[key],"request_success_total")
   if n is not None and n>=1:return snap,False
   if time.monotonic()>end:return snap,True
   guard();time.sleep(3);idx+=1
 def request(name,key,body,expected,kind):
  ownercheck();f=r/(name+".body.json");atomic_json(f,body)
  row=dict(name=name,key=key,kind=kind,at=utc(),expected_output_tokens=expected,body=ref(f),completed=False,effective_public_credit=0)
  rows.append(row);atomic_json(r/"semantic_attempts.json",rows)
  wire=r/(name+".wire");t=time.monotonic()
  try:
   rq=urllib.request.Request(url(key,"v1/chat/completions"),data=f.read_bytes(),headers={"Content-Type":"application/json","X-Request-ID":r.name+"-"+name})
   with wire.open("wb")as sink:
    with http.open(rq,timeout=240)as z:
     row["http_status"]=z.status
     while True:
      guard();b=z.read1(65536)
      if not b:break
      sink.write(b);sink.flush()
   guard();obj=json.loads(wire.read_bytes());assert row["http_status"]==200 and not obj.get("error")
   choice=obj["choices"][0];usage=obj["usage"]
   assert len(obj["choices"])==1 and choice["finish_reason"]=="length"
   assert usage["completion_tokens"]==expected and usage["total_tokens"]==usage["prompt_tokens"]+expected
   prompt=obj.get("prompt_token_ids");ids=choice.get("token_ids")
   assert isinstance(prompt,list)and len(prompt)==usage["prompt_tokens"]and all(type(x)is int for x in prompt)
   assert isinstance(ids,list)and len(ids)==expected and all(type(x)is int for x in ids)
   row.update(completed=True,usage=usage,wall_s=time.monotonic()-t,prompt_ids_sha256=hashlib.sha256(json.dumps(prompt,separators=(",",":")).encode()).hexdigest(),output_ids=ids)
   # Semantic credit is adjudicated after the independent control, not here.
   return obj
  except urllib.error.HTTPError as e:
   wire.write_bytes(e.read());row.update(http_status=e.code,error_type="HTTPError",error=str(e));raise
  except BaseException as e:row.update(error_type=type(e).__name__,error=str(e));raise
  finally:
   if wire.exists():row["wire"]=ref(wire)
   atomic_json(r/"semantic_attempts.json",rows)
 canonical=json.loads((r/"canonical_81932.body.json").read_text());canonical.pop("stream_options",None)
 canonical.update(stream=False,max_tokens=64,temperature=0,seed=20260930,ignore_eos=True,return_token_ids=True,include_reasoning=True)
 canonical.pop("kv_transfer_params",None);canonical.pop("cache_salt",None)
 try:
  # Let prior native statistics flush. No requests or service changes.
  initial=snapshot("initial");guard();time.sleep(12);baseline=snapshot("baseline")
  for n,pkey,dkey in[(0,"P0","D1"),(1,"P1","D0")]:
   prefix="semantic"+str(n);before=snapshot(prefix+"_before")
   pd_salt=r.name+"-"+prefix+"-PD";local_salt=r.name+"-"+prefix+"-LOCAL"
   pb=dict(canonical);pb.update(cache_salt=pd_salt,max_tokens=1,min_tokens=1,kv_transfer_params=dict(do_remote_decode=True,do_remote_prefill=False,remote_engine_id=None,remote_block_ids=None,remote_host=None,remote_port=None))
   p=request(prefix+"_P",pkey,pb,1,"internal_P_helper")
   kv=p.get("kv_transfer_params");assert isinstance(kv,dict)and kv.get("do_remote_prefill")and kv.get("remote_block_ids")and kv.get("remote_dcp_size")==16 and kv.get("remote_pcp_size")==1
   planned=json.loads((r/"planned_launch.json").read_text());px=next(x for x in planned if x["role"]=="P"and x["rank"]==owners[pkey]["rank"]);expected_engine=json.loads(px["argv"][px["argv"].index("--kv-transfer-config")+1])["engine_id"]
   assert kv["remote_engine_id"]==expected_engine,"producer engine identity mismatch"
   atomic_json(r/(prefix+".transfer.json"),kv)
   db=dict(canonical);db.update(cache_salt=pd_salt,kv_transfer_params=kv)
   pd=request(prefix+"_PD",dkey,db,64,"public_PD")
   pd_after,pd_lag=settle(prefix+"_PD_after",dkey,before[dkey])
   local_before=snapshot(prefix+"_local_before")
   lb=dict(canonical);lb["cache_salt"]=local_salt
   local=request(prefix+"_LOCAL",dkey,lb,64,"public_D_local_fullprefill")
   local_after,local_lag=settle(prefix+"_LOCAL_after",dkey,local_before[dkey])
   prompt_equal=pd["prompt_token_ids"]==local["prompt_token_ids"];output_equal=pd["choices"][0]["token_ids"]==local["choices"][0]["token_ids"]
   pp=p["prompt_token_ids"];dp=pd["prompt_token_ids"]
   assert pp==dp[:len(pp)]and len(dp)-len(pp)in[0,1],"P canonical prompt identity changed"
   assert prompt_equal,"D PD/local tokenized prompts differ"
   local_deltas={name:delta(local_before[dkey],local_after[dkey],name)for name in COUNTERS}
   pd_deltas={name:delta(before[dkey],pd_after[dkey],name)for name in COUNTERS}
   cold_confirmed=not local_lag and local_deltas["prefix_cache_hits_total"]==0 and local_deltas["external_prefix_cache_hits_total"]==0 and (local_deltas["prefix_cache_queries_total"]or 0)>0
   pair=dict(P=pkey,D=dkey,prompt_ids_equal=prompt_equal,output_ids_equal=output_equal,independent_cache_salts=True,local_cold_counter_confirmed=cold_confirmed,PD_counters=pd_deltas,local_counters=local_deltas,PD_metrics_lag_unknown=pd_lag,local_metrics_lag_unknown=local_lag,producer_engine_id=kv["remote_engine_id"],P_prompt_tokens=len(pp),D_prompt_tokens=len(dp),PD_output_ids=pd["choices"][0]["token_ids"],local_output_ids=local["choices"][0]["token_ids"])
   pairs.append(pair);atomic_json(r/"semantic_pairs.json",pairs)
   if not output_equal:raise RuntimeError("PD/local committed token IDs diverge; investigate raw/native paths before semantic acceptance")
   if cold_confirmed:
    rows[-1]["effective_public_credit"]=64;rows[-2]["effective_public_credit"]=64;atomic_json(r/"semantic_attempts.json",rows)
  final=snapshot("final")
  out=dict(at=utc(),measurement_valid=all(x["output_ids_equal"]and x["local_cold_counter_confirmed"]for x in pairs),inference_attempts=len(rows),completed=len(rows),native_output_tokens=sum(x["usage"]["completion_tokens"]for x in rows),effective_public_output_tokens=sum(x["effective_public_credit"]for x in rows),internal_helpers=2,pairs=pairs,limits=["Token equality of two finite canonical PD/local cases is not all-output/feature/stability/capacity proof","Different cache_salt/native miss counters isolate local control from prior prefix reuse; transfer completion and source shard logs remain audited separately","CPU source shape and zero inheritedworkspace do not certify all runtime workspace peaks","No public gateway/state/fault recovery comparison, no performance/quality rating or KEEP"])
  atomic_json(r/"semantic_summary.json",out);return out
 except BaseException as e:
  atomic_json(r/"semantic_summary.json",dict(at=utc(),measurement_valid=False,error_type=type(e).__name__,error=str(e),requests=rows,pairs=pairs,effective_public_output_tokens=sum(x["effective_public_credit"]for x in rows)));raise

