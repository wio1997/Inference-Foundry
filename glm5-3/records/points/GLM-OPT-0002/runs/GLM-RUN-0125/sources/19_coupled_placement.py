"""Opt-in physical execution-group fault domains for complete native requests.

Native KV/MTP and HTTP payloads remain owned by engines. Groups are immutable
for this gateway lifetime; a fault remains quarantined until the controller
reconciles/rebuilds native owners and starts a new gateway with a new epoch.
Logical drain/readd does not remove physical ranks or clear group faults.
"""
import math
from dataclasses import dataclass
from placement import Placement

@dataclass
class ExecutionGroup:
    key: str
    epoch: str
    members: frozenset
    faulted: bool = False

class CoupledPlacement(Placement):
    def __init__(self, replicas, policy="active_count", groups=None):
        self.execution_groups = {}
        self.member_group = {}
        configs = list(replicas)
        ids = {item["id"] for item in configs}
        for config in groups or []:
            key = config["id"]; epoch = config["epoch"]; members = config["members"]
            if not isinstance(key,str) or not key or not isinstance(epoch,str) or not epoch:
                raise ValueError("group id and native owner epoch required")
            if not isinstance(members,list) or not members or any(not isinstance(v,str) or not v for v in members):
                raise ValueError("group member ids required")
            if len(set(members)) != len(members) or not set(members) <= ids:
                raise ValueError("group members must be distinct configured endpoints")
            if key in self.execution_groups or any(v in self.member_group for v in members):
                raise ValueError("execution groups must be disjoint")
            group = ExecutionGroup(key,epoch,frozenset(members))
            self.execution_groups[key] = group
            self.member_group.update({v:group for v in members})
        super().__init__(configs,policy)

    def _add(self, config):
        super()._add(config)
        group = self.member_group.get(config["id"])
        if group is not None and group.faulted:
            self.replicas[config["id"]].unhealthy_until = math.inf

    async def release(self, lease, backend_failure=False):
        async with self.lock:
            owned = self.leases.get(lease.lease_id)
            if owned is None: return False
            if owned is not lease: raise ValueError("lease identity mismatch")
            del self.leases[lease.lease_id]
            replica = owned.replica
            replica.active.pop(lease.lease_id)
            replica.prefilling.discard(lease.lease_id)
            if backend_failure:
                group = self.member_group.get(replica.key)
                if group is None:
                    # Preserve native independent-endpoint cooldown semantics.
                    import time
                    replica.unhealthy_until = time.monotonic()+5
                else:
                    group.faulted = True
                    replica.unhealthy_until = math.inf
                    for key in group.members:
                        peer = self.replicas.get(key)
                        if peer is not None: peer.unhealthy_until = math.inf
            if replica.draining and not replica.active and self.replicas.get(replica.key) is replica:
                del self.replicas[replica.key]
            return True

    async def snapshot(self):
        import time
        async with self.lock:
            rows = []
            for r in self.replicas.values():
                group = self.member_group.get(r.key)
                rows.append({"id":r.key,"url":r.url,"generation":r.generation,
                    "active_requests":len(r.active),"prefilling_requests":len(r.prefilling),
                    "placement_policy":self.policy,
                    "reserved_output_tokens":sum(v[0] for v in r.active.values()),
                    "draining":r.draining,"decode_tps_hint":r.decode_tps,
                    "prefill_bytes_per_s_hint":r.prefill_bytes_per_s,
                    "temporarily_unhealthy":r.unhealthy_until>time.monotonic(),
                    "execution_group":None if group is None else group.key,
                    "native_owner_epoch":None if group is None else group.epoch,
                    "group_faulted":False if group is None else group.faulted})
            return rows
