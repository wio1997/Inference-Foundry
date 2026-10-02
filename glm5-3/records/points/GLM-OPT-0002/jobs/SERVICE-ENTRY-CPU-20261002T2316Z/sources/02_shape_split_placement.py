"""Opt-in input-byte domain separation for heterogeneous native replicas.

Bytes choose a preferred domain, not an estimated GPU cost or capacity.
Bound native response owners take precedence. Draining/faulted domains are
excluded; unbound requests fall back to healthy available domains.
"""
import time,uuid
from placement import Lease
from work_seconds_placement import WorkSecondsPlacement

class ShapeSplitPlacement(WorkSecondsPlacement):
    def __init__(self,replicas,policy="active_count",groups=None,fault_state_path=None,shape_config=None):
        self.shape_split=policy=="shape_split"
        self.shape_config=None
        if self.shape_split:
            if not isinstance(shape_config,dict)or set(shape_config)!={"input_threshold_bytes","prefill_members","decode_members"}:
                raise ValueError("shape split requires explicit threshold and native domains")
            threshold=shape_config["input_threshold_bytes"]
            if type(threshold)is not int or threshold<=0:raise ValueError("positive integer input threshold required")
            ids={r["id"]for r in replicas};pools=[]
            for key in["prefill_members","decode_members"]:
                members=shape_config[key]
                if not isinstance(members,list)or not members or any(not isinstance(x,str)or x not in ids for x in members)or len(set(members))!=len(members):
                    raise ValueError("unique configured native domain members required")
                pools.append(set(members))
            if pools[0]&pools[1]:raise ValueError("shape domains must be disjoint")
            self.shape_config={k:list(v)if isinstance(v,list)else v for k,v in shape_config.items()}
        super().__init__(replicas,"active_count"if self.shape_split else policy,groups,fault_state_path)
        if self.shape_split:self.policy="shape_split"

    async def acquire(self,output_budget,input_bytes):
        if not self.shape_split:return await super().acquire(output_budget,input_bytes)
        if type(output_budget)is not int or output_budget<=0 or type(input_bytes)is not int or input_bytes<0:
            raise ValueError("invalid request estimate")
        async with self.lock:
            available=[r for r in self.replicas.values()if not r.draining and r.unhealthy_until<=time.monotonic()]
            if not available:raise RuntimeError("no eligible GLM replica")
            key="prefill_members"if input_bytes>=self.shape_config["input_threshold_bytes"]else"decode_members"
            preferred=[r for r in available if r.key in self.shape_config[key]]
            eligible=preferred or available
            least=min(len(r.active)for r in eligible)
            tied=[r for r in eligible if len(r.active)==least]
            replica=tied[self.cursor%len(tied)];self.cursor+=1
            lease=Lease(uuid.uuid4().hex,replica,output_budget,input_bytes)
            replica.active[lease.lease_id]=(output_budget,input_bytes)
            self.leases[lease.lease_id]=lease;replica.prefilling.add(lease.lease_id)
            return lease

    async def snapshot(self):
        rows=await super().snapshot()
        for row in rows:
            row["shape_split_hint"]=self.shape_config if self.shape_split else None
        return rows
