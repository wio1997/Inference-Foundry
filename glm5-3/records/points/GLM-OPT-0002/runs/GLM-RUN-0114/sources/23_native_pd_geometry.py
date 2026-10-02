"""Necessary PD geometry checks for the observed GLM Mooncake implementation.

A pass is static eligibility, not live epoch, transfer or capacity evidence.
Unknown or incompatible plans must use the original native D request path.
"""
import json
MOONCAKE_SOURCE_SHA256="f30572890014d0244bf31d8e15dc5d93fb84147df37e0b2c1d1b01cbba6dd533"
def _arg(argv,key,default=None):
    found=[i for i,v in enumerate(argv)if v==key]
    if not found:return default
    if len(found)!=1 or found[0]+1>=len(argv):raise ValueError("ambiguous "+key)
    return argv[found[0]+1]
def _positive(value):
    if isinstance(value,bool):raise ValueError("boolean geometry")
    n=int(value)
    if n<=0:raise ValueError("nonpositive geometry")
    return n
def _plan(plan):
    argv=plan["argv"];env=plan.get("env",{})
    if not isinstance(argv,list)or not all(isinstance(x,str)for x in argv):raise ValueError("argv")
    sizes={name:_positive(_arg(argv,"--"+flag,1))for name,flag in
           [("tp_size","tensor-parallel-size"),("dp_size","data-parallel-size"),
            ("pp_size","pipeline-parallel-size"),("pcp_size","prefill-context-parallel-size"),
            ("dcp_size","decode-context-parallel-size")]}
    kv_arg=_arg(argv,"--kv-transfer-config");kv=json.loads(kv_arg)if kv_arg else None
    node=plan.get("node",plan.get("host"))
    if node not in {"166","167"}:raise ValueError("unknown native host")
    port=_positive(_arg(argv,"--port"))
    return dict(sizes=sizes,kv=kv,partition=env.get("VLLM_PP_LAYER_PARTITION"),
                origin="http://172.16.10."+node+":"+str(port),host="172.16.10."+node)
def assess_pd_geometry(producer,decoder,connector_source_sha256=None):
    reasons=[];out=dict(schema_version=1,eligible=False,reasons=reasons,
                       native_rule_source_sha256=MOONCAKE_SOURCE_SHA256,
                       limits=["Static intended CLI/declarations only; live source/epoch/KV metadata, receive/release and functional E2E require independent evidence"])
    if connector_source_sha256!=MOONCAKE_SOURCE_SHA256:reasons.append("native_connector_source_unknown")
    try:p=_plan(producer);d=_plan(decoder)
    except (KeyError,TypeError,ValueError)as error:
        reasons.append("native_plan_unknown");out["plan_error"]=str(error);return out
    out.update(producer=p,decoder=d)
    for name,plan,role in [("producer",p,"kv_producer"),("decoder",d,"kv_consumer")]:
        kv=plan["kv"]
        if not isinstance(kv,dict)or kv.get("kv_connector")!="MooncakeConnectorV1"or kv.get("kv_role")!=role:
            reasons.append(name+"_connector_unavailable");continue
        extra=kv.get("kv_connector_extra_config",{})
        if not isinstance(extra,dict):reasons.append(name+"_connector_declaration_unknown");continue
        for side,want in [("prefill",p["sizes"]),("decode",d["sizes"])]:
            declaration=extra.get(side)
            if not isinstance(declaration,dict):reasons.append(name+"_"+side+"_declaration_unknown");continue
            for key in ["tp_size","dp_size","pp_size"]:
                declared=declaration.get(key,1 if key=="pp_size"else None)
                if type(declared)is not int or declared!=want[key]:
                    reasons.append(name+"_"+side+"_"+key+"_mismatch")
        if p["partition"] is not None and extra.get("prefill",{}).get("pp_layer_partition")!=p["partition"]:reasons.append(name+"_prefill_layer_partition_mismatch")
        if plan["sizes"]["pp_size"]>1 and plan["sizes"]["pcp_size"]>1:reasons.append(name+"_pp_pcp_mutex")
    if d["sizes"]["pp_size"]!=1:reasons.append("native_decode_pp_must_be_1")
    if p["sizes"]["tp_size"]<d["sizes"]["tp_size"]:reasons.append("native_prefill_tp_smaller_than_decode")
    remote=p["sizes"]["pcp_size"]*p["sizes"]["dcp_size"];local=d["sizes"]["pcp_size"]*d["sizes"]["dcp_size"]
    if remote<local or remote%local:reasons.append("native_remote_local_cp_not_divisible")
    # This implementation has only been scoped to the independently owned DP1 class.
    if p["sizes"]["dp_size"]!=1 or d["sizes"]["dp_size"]!=1:reasons.append("global_dp_geometry_not_verified")
    out["eligible"]=not reasons
    return out
