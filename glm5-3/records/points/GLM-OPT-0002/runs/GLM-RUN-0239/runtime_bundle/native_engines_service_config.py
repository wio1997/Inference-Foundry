"""Compile one or two independent GLM native engine fault domains.

Each engine delegates its root, CLI geometry and complete local worker validation
to standalone_service_config. This is a recorded-evidence compiler: a host
controller must check all live identities before publishing the configuration.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from standalone_service_config import compile_config as compile_engine

EVIDENCE_NAME = "native_engines_resident.json"


def _artifact(reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "bytes", "sha256"}:
        raise ValueError("complete work hint artifact reference required")
    path = reference["path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("work hint evidence path must be absolute")
    if type(reference["bytes"]) is not int or reference["bytes"] <= 0:
        raise ValueError("invalid work hint artifact size")
    sha = reference["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("invalid work hint artifact hash")
    raw = Path(path).read_bytes()
    if len(raw) != reference["bytes"] or hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("work hint evidence changed")
    return raw

def _routing_hint(reference, epoch):
    hint = json.loads(_artifact(reference))
    fields = {"schema_version", "native_owner_epoch", "cohort_id", "decode_tps",
              "prefill_bytes_per_s", "method", "sources", "limitations"}
    if not isinstance(hint, dict) or set(hint) != fields or type(hint["schema_version"]) is not int or hint["schema_version"] != 1:
        raise ValueError("unrecognized observed work hint schema")
    if hint["native_owner_epoch"] != epoch:
        raise ValueError("observed work hint belongs to another native owner epoch")
    for key in ("cohort_id", "method"):
        if not isinstance(hint[key], str) or not hint[key]:
            raise ValueError("observed work hint provenance is required")
    for key in ("decode_tps", "prefill_bytes_per_s"):
        value = hint[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("observed work rates must be positive finite numbers")
    if not isinstance(hint["sources"], list) or not hint["sources"]:
        raise ValueError("observed work hint requires source artifacts")
    for source in hint["sources"]:
        _artifact(source)
    if not isinstance(hint["limitations"], list) or not hint["limitations"] or not all(isinstance(v,str)and v for v in hint["limitations"]):
        raise ValueError("observed work hint limitations are required")
    return hint

def compile_config(material, state_dir):
    if not isinstance(material, dict) or set(material) not in ({"engines", "placement"},{"engines", "placement", "compatibility"}):
        raise ValueError("engines and placement evidence are required")
    engines = material["engines"]
    if not isinstance(engines, list) or not 1 <= len(engines) <= 2:
        raise ValueError("one or two task-native engines are required")
    replicas, groups, geometry = [], [], {}
    hints = {}
    ids, replica_ids, hosts = set(), set(), set()
    environment = None
    for engine in sorted(engines, key=lambda value: value["id"]):
        if set(engine) not in ({"id", "replica_id", "plans", "roots", "members"},
                               {"id", "replica_id", "plans", "roots", "members", "routing_hint"}):
            raise ValueError("unrecognized engine evidence fields")
        ident, replica = engine["id"], engine["replica_id"]
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", ident):
            raise ValueError("invalid native engine domain ID")
        if not isinstance(replica, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", replica):
            raise ValueError("invalid API replica ID")
        if ident in ids or replica in replica_ids:
            raise ValueError("engine and replica IDs must be unique")
        ids.add(ident)
        replica_ids.add(replica)
        config = compile_engine(engine["plans"], engine["roots"],
                                engine["members"], state_dir)
        engine_hosts = {plan["host"] for plan in engine["plans"].values()}
        if hosts & engine_hosts:
            raise ValueError("task hosts cannot belong to different native engines")
        hosts |= engine_hosts
        geometry[ident] = config["geometry"]
        native_replica = json.loads(config["environment"]["GLM_REPLICAS"])[0]
        row = dict(id=replica, url=native_replica["url"])
        epoch = config["native_domains"][0]["epoch"]
        if "routing_hint" in engine:
            if not isinstance(material["placement"], dict) or material["placement"].get("kind") != "work_seconds":
                raise ValueError("observed rates require explicit work-seconds placement")
            hint = _routing_hint(engine["routing_hint"], epoch)
            row.update(decode_tps=hint["decode_tps"], prefill_bytes_per_s=hint["prefill_bytes_per_s"])
            hints[replica] = dict(artifact=engine["routing_hint"], observation=hint)
        replicas.append(row)
        groups.append(dict(id=ident, epoch=epoch, members=[replica]))
        if environment is None:
            environment = dict(config["environment"])
    compatibility = None
    if "compatibility" in material:
        rule = material["compatibility"]
        if not isinstance(rule,dict) or set(rule)!={"kind","sources"} or rule["kind"]!="native_v1":
            raise ValueError("explicit native V1 compatibility evidence required")
        if not isinstance(rule["sources"],list) or not rule["sources"]:
            raise ValueError("compatibility evidence sources required")
        for source in rule["sources"]:
            _artifact(source)
        members = [engine["replica_id"] for engine in engines
                   if all(plan["environment"].get("VLLM_USE_V2_MODEL_RUNNER")!="1"
                          for plan in engine["plans"].values())]
        if not members:
            raise ValueError("resident native V1 compatibility domain required")
        if material["placement"]["kind"] not in ("shape_split","shape_split_idle_spill"):
            raise ValueError("compatibility prototype requires explicit shape placement")
        membership = {key:sorted(members)for key in ("structured_output","thinking_budget")}
        environment["GLM_COMPATIBILITY_MEMBERS"] = json.dumps(membership)
        compatibility = dict(kind="native_v1",members=membership,sources=rule["sources"],
                             ownership="bound owners remain fixed; incompatible owner fails before RPC")
    placement = material["placement"]
    if not isinstance(placement, dict) or "kind" not in placement:
        raise ValueError("explicit native API placement is required")
    kind = placement["kind"]
    if kind in ("active_count", "work_seconds"):
        if set(placement) != {"kind"}:
            raise ValueError("unexpected native placement fields")
    elif kind in ("shape_split", "shape_split_idle_spill"):
        if set(placement) != {"kind", "input_threshold_bytes",
                              "prefill_members", "decode_members"}:
            raise ValueError("complete shape placement is required")
        threshold = placement["input_threshold_bytes"]
        if type(threshold) is not int or threshold <= 0:
            raise ValueError("shape threshold must be a positive integer")
        prefill, decode = placement["prefill_members"], placement["decode_members"]
        for members in (prefill, decode):
            if not isinstance(members, list) or not members or not all(
                    isinstance(member, str) for member in members):
                raise ValueError("nonempty native replica sets are required")
            if len(set(members)) != len(members):
                raise ValueError("duplicate native shape members")
        if set(prefill) & set(decode) or set(prefill) | set(decode) != replica_ids:
            raise ValueError("shape members must partition recorded native replicas")
        environment["GLM_SHAPE_SPLIT"] = json.dumps(
            {key: value for key, value in placement.items() if key != "kind"})
    else:
        raise ValueError("unsupported native-engine placement policy")
    environment.update(GLM_REPLICAS=json.dumps(replicas),
                       GLM_EXECUTION_GROUPS=json.dumps(groups),
                       GLM_PLACEMENT_POLICY=kind,
                       GLM_PD_PRODUCERS="{}", GLM_PD_NATIVE_PLANS="{}")
    result = dict(schema_version=1, deployment="independent_native_engines",
                engine_geometry=geometry, placement=placement,
                environment=environment, native_domains=groups,
                limitations=[
                    "Recorded native root and worker identities require fresh host controller checks",
                    "Each coupled engine is one native STORE domain; independent engines keep separate stores",
                    "Native engine replacement retires its own epoch; STORE is not replicated",
                    "No PD helper or native KV transfer is configured",
                    "Shape placement uses input bytes and preserves native owner affinity",
                    "Configuration does not certify correctness, stable SLO capacity or hardware bounds",
                ])

    if compatibility is not None:
        result["compatibility"] = compatibility
        result["limitations"] += [
            "V1 compatibility eligibility filters precede shape/load selection",
            "Bodies/native validation/bytes are unchanged; incompatible native owner fails before RPC",
            "Native V2 grammar failure remains unresolved; compatibility costs require real E2E",
        ]

    if kind == "work_seconds":
        result["routing_hints"] = hints
        result["limitations"] += [
            "Work hints bind to recorded native owner epochs and hashed source artifacts",
            "Missing hints on any eligible engine fall back to counts for all",
            "Observed rates do not infer progress, batch speedup or native background work",
            "Work hints do not certify scheduling optimality or stable capacity",
        ]
    return result


def render(resident_dir, state_dir):
    path = Path(resident_dir) / EVIDENCE_NAME
    raw = path.read_bytes()
    config = compile_config(json.loads(raw), state_dir)
    config["resident_evidence"] = {
        EVIDENCE_NAME: dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())}
    return config


def checked_config(path):
    config = json.loads(Path(path).read_text())
    evidence = config.get("resident_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {EVIDENCE_NAME}:
        raise ValueError("complete native engine resident evidence is required")
    item = evidence[EVIDENCE_NAME]
    raw = Path(item["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("native engine resident evidence changed")
    state_dir = Path(config["environment"]["GLM_RESPONSE_OWNER_STATE_PATH"]).parent
    expected = compile_config(json.loads(raw), state_dir)
    if set(config) != set(expected) | {"resident_evidence"} or any(
            config.get(key) != value for key, value in expected.items()):
        raise ValueError("service config disagrees with native engine evidence")
    return config, state_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resident-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(render(args.resident_dir, args.state_dir),
                                     indent=2) + "\n")


if __name__ == "__main__":
    main()
