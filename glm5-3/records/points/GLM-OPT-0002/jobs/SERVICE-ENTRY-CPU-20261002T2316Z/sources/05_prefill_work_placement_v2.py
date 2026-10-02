"""Opt-in prefill byte backlog ordering; native engines retain all inference state.

Bytes are a workload hint, not GPU time or capacity. A native first output ends
the observed prefill phase. No token/sampling/HTTP payload changes are made.
Native background work beyond HTTP leases is not represented here.
"""
import time,uuid
from placement import Lease
from response_affinity import ResponseAffinityPlacement

class PrefillWorkPlacement(ResponseAffinityPlacement):
    def __init__(self,replicas,policy="active_count",groups=None,fault_state_path=None):
        self.use_prefill_bytes=policy=="prefill_bytes"
        super().__init__(replicas,"active_count" if self.use_prefill_bytes else policy,groups,fault_state_path)
        if self.use_prefill_bytes:self.policy="prefill_bytes"

    async def acquire(self,output_budget,input_bytes):
        if not self.use_prefill_bytes:return await super().acquire(output_budget,input_bytes)
        if type(output_budget)is not int or output_budget<=0 or type(input_bytes)is not int or input_bytes<0:
            raise ValueError("invalid request estimate")
        async with self.lock:
            available=[r for r in self.replicas.values()if not r.draining and r.unhealthy_until<=time.monotonic()]
            if not available:raise RuntimeError("no eligible GLM replica")
            calibrated=all(r.decode_tps is not None for r in available)
            scores=[(sum(r.active[key][1]for key in r.prefilling if key in r.active),
                     r.score() if calibrated else len(r.active))for r in available]
            low=min(scores);tied=[r for r,s in zip(available,scores)if s==low]
            replica=tied[self.cursor%len(tied)];self.cursor+=1
            lease=Lease(uuid.uuid4().hex,replica,output_budget,input_bytes)
            replica.active[lease.lease_id]=(output_budget,input_bytes)
            self.leases[lease.lease_id]=lease;replica.prefilling.add(lease.lease_id)
            return lease

    async def acquire_for_owner(self,output_budget,input_bytes,owner=None):
        # The gateway uses this entry for every request; no prior owner uses
        # the current routing policy, while a bound native owner stays fixed.
        if owner is None:return await self.acquire(output_budget,input_bytes)
        return await super().acquire_for_owner(output_budget,input_bytes,owner)

    async def snapshot(self):
        rows=await super().snapshot()
        async with self.lock:
            for row in rows:
                r=self.replicas.get(row["id"])
                # Concurrent logical replacement makes this observation unknown.
                same=r is not None and r.generation==row["generation"]and r.url==row["url"]
                row["prefill_input_bytes_hint"]=(sum(r.active[k][1]for k in r.prefilling if k in r.active)if same else None)
        return rows
