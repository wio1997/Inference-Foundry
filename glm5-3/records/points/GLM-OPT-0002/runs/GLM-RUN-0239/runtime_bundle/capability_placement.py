"""Opt-in native V1 compatibility admission; never rewrites or replays requests."""
import json,time,uuid
from placement import Lease
from shape_split_placement import ShapeSplitPlacement

FEATURES=frozenset({"structured_output","thinking_budget"})

def request_features(path,method,body):
    if method!="POST" or path not in {"/v1/chat/completions","/v1/completions","/v1/responses"}:
        return frozenset()
    try: value=json.loads(body)
    except (ValueError,UnicodeError): return frozenset()
    if not isinstance(value,dict): return frozenset()
    required=set()
    choice=value.get("tool_choice")
    if choice=="required" or isinstance(choice,dict):
        required.add("structured_output")
    fmt=value.get("response_format")
    if isinstance(fmt,dict) and fmt.get("type") in {"json_object","json_schema","structural_tag"}:
        required.add("structured_output")
    structured=value.get("structured_outputs")
    if structured is not None:
        required.add("structured_output")
    text=value.get("text")
    if isinstance(text,dict) and isinstance(text.get("format"),dict) and text["format"].get("type") in {"json_object","json_schema","structural_tag"}:
        required.add("structured_output")
    if "thinking_token_budget" in value:
        required.add("thinking_budget")
    return frozenset(required)

class CapabilityPlacement(ShapeSplitPlacement):
    def __init__(self,replicas,policy="active_count",groups=None,fault_state_path=None,shape_config=None,compatibility_config=None):
        super().__init__(replicas,policy,groups,fault_state_path,shape_config)
        self.compatibility=None
        if compatibility_config is not None:
            if not self.shape_split:
                raise ValueError("compatibility prototype requires explicit shape placement")
            if not isinstance(compatibility_config,dict) or set(compatibility_config)!=FEATURES:
                raise ValueError("complete compatibility feature membership required")
            ids=set(self.replicas)
            for members in compatibility_config.values():
                if not isinstance(members,list) or not members or len(set(members))!=len(members) or any(not isinstance(m,str)or m not in ids for m in members):
                    raise ValueError("compatible members must be unique declared replicas")
            self.compatibility={k:frozenset(v)for k,v in compatibility_config.items()}

    async def acquire_request(self,output_budget,input_bytes,owner,path,method,body):
        features=request_features(path,method,body) if self.compatibility is not None else frozenset()
        if not features:
            return await self.acquire_for_owner(output_budget,input_bytes,owner)
        allowed=set(self.replicas)
        for feature in features:
            allowed.intersection_update(self.compatibility[feature])
        if owner is not None:
            if owner["replica"] not in allowed:
                raise RuntimeError("native response owner lacks required compatibility; no cross-owner replay")
            return await self.acquire_for_owner(output_budget,input_bytes,owner)
        if type(output_budget)is not int or output_budget<=0 or type(input_bytes)is not int or input_bytes<0:
            raise ValueError("invalid request estimate")
        async with self.lock:
            available=[r for r in self.replicas.values()if r.key in allowed and not r.draining and r.unhealthy_until<=time.monotonic()]
            if not available:
                raise RuntimeError("no eligible compatible GLM replica")
            key="prefill_members"if input_bytes>=self.shape_config["input_threshold_bytes"]else"decode_members"
            preferred=[r for r in available if r.key in self.shape_config[key]]
            eligible=preferred or available
            if self.idle_spill and key=="decode_members" and preferred and all(r.active for r in preferred):
                peers=[r for r in available if r.key in self.shape_config["prefill_members"] and not r.active and not r.prefilling]
                if peers: eligible=peers
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
            row["compatibility_features"]=sorted(k for k,v in self.compatibility.items()if row["id"]in v)if self.compatibility is not None else None
        return rows
