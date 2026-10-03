"""Render the GLM gateway configuration from recorded native process domains.

This performs no process, model, network, or request operation. The controller
must recheck live boot/start/argv, NPU ancestry and idle before starting a service.
The gateway keeps native Responses ownership and rejects incompatible PD geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path

CONNECTOR_SOURCE_SHA256 = "f30572890014d0244bf31d8e15dc5d93fb84147df37e0b2c1d1b01cbba6dd533"


def compile_config(plans, owners, members, state_dir, input_threshold_bytes=8192):
    state_dir = Path(state_dir)
    if not state_dir.is_absolute():
        raise ValueError("service state directory must be absolute")
    if type(input_threshold_bytes) is not int or input_threshold_bytes <= 0:
        raise ValueError("input threshold must be a positive integer")
    if set(owners) != {"D0", "D1"} or set(members) != set(owners):
        raise ValueError("this two-node service requires both recorded native domains")
    roles = {}
    replicas = []
    groups = []
    all_workers = set()
    for key, owner in sorted(owners.items()):
        matches = [plan for plan in plans if plan["node"] == owner["host"]]
        if len(matches) != 1:
            raise ValueError("one native launch plan per host is required")
        plan = matches[0]
        if owner["argv"] != plan["argv"] or owner["port"] != plan["port"]:
            raise ValueError("native API does not match its recorded launch plan")
        if owner["rank"] != plan["rank"] or owner["role"] != plan["role"]:
            raise ValueError("native API alias/role does not match its launch plan")
        if owner["native_dp_size"] != 1:
            raise ValueError("this service groups separate DP1 native instances only")
        member = members[key]
        if not member["API_alive"] or member["API_owner"] != owner:
            raise ValueError("native member snapshot does not match its API owner")
        workers = member["npu_worker_pids"]
        if len(workers) != 16 or len(set(workers)) != 16:
            raise ValueError("each current native domain must contain sixteen NPU workers")
        physical_workers = {(owner["host"], pid) for pid in workers}
        if all_workers & physical_workers:
            raise ValueError("native execution groups must not overlap")
        all_workers |= physical_workers
        targets = {row["pid"]: row for row in member["owned_targets"]}
        if not set(workers) <= set(targets):
            raise ValueError("NPU workers must belong to the recorded API ancestry")
        native = sorted(
            [dict(pid=pid, identity=targets[pid]["identity"]) for pid in workers],
            key=lambda row: row["pid"],
        )
        for identity in [owner["identity"]] + [row["identity"] for row in native]:
            if not identity["boot_id"] or not str(identity["start_ticks"]).isdigit():
                raise ValueError("native domains require boot and start identities")
        epoch = hashlib.sha256(json.dumps(
            dict(owner=owner, native_NPU16=native),
            sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        url = "http://172.16.10." + owner["host"] + ":" + str(owner["port"])
        replicas.append(dict(id=key, url=url))
        groups.append(dict(id=key + "-native", epoch=epoch, members=[key]))
        if owner["role"] in roles:
            raise ValueError("one P and one D role are required")
        roles[owner["role"]] = (key, url, plan)
    if set(roles) != {"P", "D"}:
        raise ValueError("one P and one D role are required")
    pkey, purl, producer = roles["P"]
    dkey, durl, decoder = roles["D"]
    producer_args = producer["argv"]
    if "--kv-transfer-config" not in producer_args:
        raise ValueError("declared producer requires native KV configuration")
    kv = json.loads(producer_args[producer_args.index("--kv-transfer-config") + 1])
    if kv["kv_connector"] != "MooncakeConnectorV1" or kv["kv_role"] != "kv_producer":
        raise ValueError("only the recorded native Mooncake producer is supported")
    producers = {durl: dict(url=purl, remote_host="172.16.10." + producer["node"],
                           remote_port=int(kv["kv_port"]),
                           dcp_size=int(producer_args[producer_args.index(
                               "--decode-context-parallel-size") + 1]))}
    native_plans = {durl: dict(producer=producer, decoder=decoder,
                              connector_source_sha256=CONNECTOR_SOURCE_SHA256)}
    environment = {
        "GLM_REPLICAS": json.dumps(replicas),
        "GLM_EXECUTION_GROUPS": json.dumps(groups),
        "GLM_PLACEMENT_POLICY": "shape_split",
        "GLM_SHAPE_SPLIT": json.dumps(dict(
            input_threshold_bytes=input_threshold_bytes,
            prefill_members=[dkey], decode_members=[pkey])),
        "GLM_PD_PRODUCERS": json.dumps(producers),
        "GLM_PD_NATIVE_PLANS": json.dumps(native_plans),
        "GLM_RESPONSE_OWNER_STATE_PATH": str(state_dir / "response_owners.json"),
        "GLM_ROUTER_TRACE_PATH": str(state_dir / "router_trace.jsonl"),
        "GLM_ROUTER_AUDIT_DIR": str(state_dir / "router_wire"),
        "GLM_PD_AUDIT_DIR": str(state_dir / "native_PD_raw"),
    }
    return dict(schema_version=1, environment=environment,
                native_domains=groups,
                limitations=[
                    "Recorded identities require a fresh controller live check before launch",
                    "Physical native API restart retires its native response store; no replication",
                    "Shape placement is a byte hint, not GPU cost or native background occupancy",
                    "PD geometry is checked by native_pd_transport_v3 before any producer helper",
                    "One gateway process; no stable SLO capacity or KEEP is certified",
                ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resident-dir", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.resident_dir
    config = compile_config(
        json.loads((source / "planned_launch.json").read_text()),
        json.loads((source / "adopted_model_identities.json").read_text()),
        json.loads((source / "native_member_identities.json").read_text()),
        args.state_dir)
    config["resident_evidence"] = {
        name: dict(path=str(source / name), sha256=hashlib.sha256(
            (source / name).read_bytes()).hexdigest())
        for name in ["planned_launch.json", "adopted_model_identities.json",
                     "native_member_identities.json"]
    }
    args.output.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()

