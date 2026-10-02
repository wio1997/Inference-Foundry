"""Observed work hints for unbound routing, not device time or capacity bounds.

Pending prefill bytes and full output reservations share estimated seconds.
Rates are optional cohort-specific hints. Missing rates on any eligible member
fall back to counts for all, without comparing counts to seconds. No progress,
batch speedup, shared-EP interference, or native background tail is inferred.
"""
import math,time,uuid
from placement import Lease
from prefill_work_placement_v2 import PrefillWorkPlacement

class WorkSecondsPlacement(PrefillWorkPlacement):
    def __init__(self,replicas,policy="active_count",groups=None,fault_state_path=None):
        self.use_work_seconds=policy=="work_seconds"
        super().__init__(replicas,"active_count" if self.use_work_seconds else policy,groups,fault_state_path)
        if self.use_work_seconds:
            self.policy="work_seconds"

    @staticmethod
    def rates_known(replica):
        return all(type(v)in(int,float)and math.isfinite(v)and v>0
                   for v in[replica.decode_tps,replica.prefill_bytes_per_s])

    @staticmethod
    def reserved_seconds(replica):
        decode=sum(v[0]for v in replica.active.values())/replica.decode_tps
        prefill=sum(replica.active[k][1]for k in replica.prefilling if k in replica.active)/replica.prefill_bytes_per_s
        return decode+prefill

    async def acquire(self,output_budget,input_bytes):
        if not self.use_work_seconds:return await super().acquire(output_budget,input_bytes)
        if type(output_budget)is not int or output_budget<=0 or type(input_bytes)is not int or input_bytes<0:
            raise ValueError("invalid request estimate")
        async with self.lock:
            available=[r for r in self.replicas.values()if not r.draining and r.unhealthy_until<=time.monotonic()]
            if not available:raise RuntimeError("no eligible GLM replica")
            calibrated=all(self.rates_known(r)for r in available)
            scores=[(self.reserved_seconds(r)+output_budget/r.decode_tps+input_bytes/r.prefill_bytes_per_s,len(r.active))
                    if calibrated else (len(r.active),)for r in available]
            low=min(scores);tied=[r for r,s in zip(available,scores)if s==low]
            replica=tied[self.cursor%len(tied)];self.cursor+=1
            lease=Lease(uuid.uuid4().hex,replica,output_budget,input_bytes)
            replica.active[lease.lease_id]=(output_budget,input_bytes)
            self.leases[lease.lease_id]=lease;replica.prefilling.add(lease.lease_id)
            return lease

    async def snapshot(self):
        rows=await super().snapshot()
        async with self.lock:
            available=[r for r in self.replicas.values()if not r.draining and r.unhealthy_until<=time.monotonic()]
            calibrated=bool(available)and all(self.rates_known(r)for r in available)
            for row in rows:
                r=self.replicas.get(row["id"])
                same=r is not None and r.generation==row["generation"]and r.url==row["url"]
                row["reserved_work_seconds_hint"]=self.reserved_seconds(r)if same and self.rates_known(r)else None
                row["work_ranking_calibrated"]=calibrated if self.use_work_seconds else None
        return rows
