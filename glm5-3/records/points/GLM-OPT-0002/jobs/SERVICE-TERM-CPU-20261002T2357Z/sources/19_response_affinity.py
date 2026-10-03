"""Native Responses store owner affinity; IDs point to API process domains, not replicated state."""
import asyncio,json,os,fcntl,time,uuid
from pathlib import Path
from placement import Lease
from persistent_coupled_placement import PersistentCoupledPlacement
class ResponseOwnerIndex:
 def __init__(self,placement,path=None):
  self.placement=placement;self.entries={};self.error=None;self.fd=None;self.path=None
  if path is None:return
  p=Path(path)
  if not p.is_absolute()or not p.parent.is_dir()or p.is_symlink():raise ValueError("existing absolute owner journal parent required")
  fd=os.open(str(p)+".lock",os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
  try:
   fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
   prior=json.loads(p.read_text())if p.exists()else {"schema":1,"owners":{}}
   if not isinstance(prior,dict)or prior.get("schema")!=1 or not isinstance(prior.get("owners"),dict):raise ValueError("invalid response owner journal")
   for key,row in prior["owners"].items():
    if not isinstance(key,str)or not key or not isinstance(row,dict)or set(row)!={"replica","url","group","epoch"}or any(not isinstance(x,str)or not x for x in row.values()):raise ValueError("invalid response owner entry")
   self.entries=prior["owners"];self.fd=fd;self.path=p
  except BaseException:os.close(fd);raise
 def resolve(self,response_id):
  row=self.entries.get(response_id)
  return dict(row)if row is not None else None
 def bind(self,response_id,lease):
  if not isinstance(response_id,str)or not response_id:raise ValueError("native response id required")
  if self.error:raise RuntimeError("response owner journal unavailable")
  group=self.placement.member_group.get(lease.replica.key)
  if group is None:raise RuntimeError("stateful response requires declared native owner epoch")
  row={"replica":lease.replica.key,"url":lease.replica.url,"group":group.key,"epoch":group.epoch}
  if self.entries.get(response_id)==row:return False
  entries=dict(self.entries);entries[response_id]=row
  if self.path is not None:
   temp=self.path.with_name(self.path.name+"."+uuid.uuid4().hex+".tmp")
   try:
    fd=os.open(temp,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"w")as out:
     json.dump({"schema":1,"owners":entries},out,separators=(",",":"));out.flush();os.fsync(out.fileno())
    os.replace(temp,self.path)
    d=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(d)
    finally:os.close(d)
   except BaseException as e:
    self.error=str(e)
    try:temp.unlink()
    except FileNotFoundError:pass
    raise
  self.entries=entries;return True
 def close(self):
  if self.fd is not None:os.close(self.fd);self.fd=None

class ResponseAffinityPlacement(PersistentCoupledPlacement):
 async def acquire_for_owner(self,output_budget,input_bytes,owner=None):
  if owner is None:return await super().acquire(output_budget,input_bytes)
  if type(output_budget)is not int or output_budget<=0 or type(input_bytes)is not int or input_bytes<0:raise ValueError("invalid request estimate")
  async with self.lock:
   replica=self.replicas.get(owner["replica"]);group=self.member_group.get(owner["replica"])
   if replica is None or group is None or replica.url!=owner["url"]or group.key!=owner["group"]or group.epoch!=owner["epoch"]or group.faulted or replica.draining or replica.unhealthy_until>time.monotonic():
    raise RuntimeError("native response owner unavailable; no cross-owner replay")
   lease=Lease(uuid.uuid4().hex,replica,output_budget,input_bytes);replica.active[lease.lease_id]=(output_budget,input_bytes);self.leases[lease.lease_id]=lease;replica.prefilling.add(lease.lease_id);return lease

def request_owner_key(path,method,body):
 if path=="/v1/responses"and method=="POST":
  try:value=json.loads(body)
  except (ValueError,UnicodeError):return None
  previous=(value.get("previous_response_id") or value.get("request_id"))if isinstance(value,dict)else None
  return previous if isinstance(previous,str)and previous else None
 if path.startswith("/v1/responses/"):
  tail=path[len("/v1/responses/"):]
  if method=="GET"and "/"not in tail:return tail or None
  if method=="POST"and tail.endswith("/cancel")and "/"not in tail[:-7]:return tail[:-7]or None
 return None

class ResponsesOwnerObserver:
 """Metadata only. Native bytes, IDs and sampling bodies are never rewritten."""
 def __init__(self):
  import codecs
  self.decoder=codecs.getincrementaldecoder("utf-8")();self.pending="";self.data=[];self.event=None;self.response_id=None;self.error=False
 def feed(self,block):
  ids=[];self.pending+=self.decoder.decode(block)
  while "\n"in self.pending:
   line,self.pending=self.pending.split("\n",1);line=line.rstrip("\r")
   if not line:
    if self.data:
     raw="\n".join(self.data)
     if raw!="[DONE]":
      obj=json.loads(raw);typ=obj.get("type");response=obj.get("response")
      if isinstance(response,dict)and typ in {"response.created","response.in_progress","response.completed","response.incomplete","response.failed","response.cancelled"}:
       ident=response.get("id")
       if isinstance(ident,str)and ident:
        if self.response_id is not None and self.response_id!=ident:raise ValueError("native stream changed response id")
        if self.response_id is None:ids.append(ident);self.response_id=ident
       error=response.get("error")
       if isinstance(error,dict)and error.get("type")=="server_error":self.error=True
      error=obj.get("error")
      if isinstance(error,dict)and error.get("type")=="server_error":self.error=True
    self.data=[];self.event=None
   elif line.startswith("data:"):
    value=line[5:];self.data.append(value[1:]if value.startswith(" ")else value)
   elif line.startswith("event:"):self.event=line[6:].strip()
  return ids
