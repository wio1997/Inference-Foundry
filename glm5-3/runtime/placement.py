"""Request leases and resource-aware placement for complete GLM replicas.

No device/KV state is invented here. Each engine retains its native state owner.
Uncalibrated replicas use active-request counts, not fabricated token capacity.
"""
import asyncio,json,math,time,uuid
from dataclasses import dataclass,field
from typing import Optional

@dataclass
class Replica:
    key:str
    url:str
    generation:int
    decode_tps:Optional[float]=None
    prefill_bytes_per_s:Optional[float]=None
    draining:bool=False
    active:dict=field(default_factory=dict)
    unhealthy_until:float=0
    def score(self):
        if self.decode_tps is None:return float(len(self.active))
        return sum(output/self.decode_tps+(size/self.prefill_bytes_per_s if self.prefill_bytes_per_s else 0) for output,size in self.active.values())

@dataclass(frozen=True)
class Lease:
    lease_id:str
    replica:Replica
    output_budget:int
    input_bytes:int

class Placement:
    def __init__(self,replicas):
        self.lock=asyncio.Lock();self.replicas={};self.leases={};self.generation=0;self.cursor=0
        for config in replicas:self._add(config)
    def _add(self,config):
        key=config['id'];url=config['url'].rstrip('/')
        if key in self.replicas:raise ValueError('replica id already present or draining')
        if not url.startswith(('http://','https://')):raise ValueError('absolute HTTP replica URL required')
        def rate(name):
            value=config.get(name)
            if value is not None and (not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0):raise ValueError('rate must be positive finite or unknown')
            return value
        self.generation+=1;self.replicas[key]=Replica(key,url,self.generation,rate('decode_tps'),rate('prefill_bytes_per_s'))
    async def add(self,config):
        async with self.lock:self._add(config)
    async def remove(self,key):
        async with self.lock:
            replica=self.replicas[key];replica.draining=True
            if not replica.active:del self.replicas[key]
    async def acquire(self,output_budget,input_bytes):
        if type(output_budget) is not int or output_budget<=0 or type(input_bytes) is not int or input_bytes<0:raise ValueError('invalid request estimate')
        async with self.lock:
            available=[r for r in self.replicas.values() if not r.draining and r.unhealthy_until<=time.monotonic()]
            if not available:raise RuntimeError('no eligible GLM replica')
            # Comparing calibrated seconds to uncalibrated counts is invalid.
            calibrated=all(r.decode_tps is not None for r in available)
            scores=[r.score() if calibrated else len(r.active) for r in available];low=min(scores)
            tied=[r for r,s in zip(available,scores) if s==low];replica=tied[self.cursor%len(tied)];self.cursor+=1
            lease=Lease(uuid.uuid4().hex,replica,int(output_budget),int(input_bytes));replica.active[lease.lease_id]=(lease.output_budget,lease.input_bytes);self.leases[lease.lease_id]=lease;return lease
    async def release(self,lease,backend_failure=False):
        async with self.lock:
            owned=self.leases.get(lease.lease_id)
            if owned is None:return False
            if owned is not lease:raise ValueError('lease identity mismatch')
            del self.leases[lease.lease_id]
            replica=owned.replica;replica.active.pop(lease.lease_id)
            if backend_failure:replica.unhealthy_until=time.monotonic()+5
            if replica.draining and not replica.active and self.replicas.get(replica.key) is replica:del self.replicas[replica.key]
            return True
    async def snapshot(self):
        async with self.lock:return [{'id':r.key,'url':r.url,'generation':r.generation,'active_requests':len(r.active),'reserved_output_tokens':sum(v[0] for v in r.active.values()),'draining':r.draining,'decode_tps_hint':r.decode_tps,'prefill_bytes_per_s_hint':r.prefill_bytes_per_s,'temporarily_unhealthy':r.unhealthy_until>time.monotonic()} for r in self.replicas.values()]

def estimate_request(body):
    """Scheduling hint only; unsupported/invalid JSON is forwarded unchanged."""
    try:
        request=json.loads(body);budget=request.get('max_completion_tokens',request.get('max_tokens',16));choices=max(request.get('n',1),request.get('best_of',1),1)
        if type(budget) is not int or budget<=0 or type(choices) is not int:raise ValueError()
        return budget*choices,len(body)
    except (ValueError,TypeError,AttributeError):return 16,len(body)
