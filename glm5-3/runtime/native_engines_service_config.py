"""Compile one or two independent GLM native engine fault domains.

Each engine delegates its root, CLI geometry and complete local worker validation
to standalone_service_config. This is a recorded-evidence compiler: a host
controller must check all live identities before publishing the configuration.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from standalone_service_config import compile_config as compile_engine

EVIDENCE_NAME = "native_engines_resident.json"


def compile_config(material, state_dir):
    if not isinstance(material, dict) or set(material) != {"engines", "placement"}:
        raise ValueError("engines and placement evidence are required")
    engines = material["engines"]
    if not isinstance(engines, list) or not 1 <= len(engines) <= 2:
        raise ValueError("one or two task-native engines are required")
    replicas, groups, geometry = [], [], {}
    ids, replica_ids, hosts = set(), set(), set()
    environment = None
    for engine in sorted(engines, key=lambda value: value["id"]):
        if set(engine) != {"id", "replica_id", "plans", "roots", "members"}:
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
        replicas.append(dict(id=replica, url=native_replica["url"]))
        groups.append(dict(id=ident, epoch=config["native_domains"][0]["epoch"],
                           members=[replica]))
        if environment is None:
            environment = dict(config["environment"])
    placement = material["placement"]
    if not isinstance(placement, dict) or "kind" not in placement:
        raise ValueError("explicit native API placement is required")
    kind = placement["kind"]
    if kind == "active_count":
        if set(placement) != {"kind"}:
            raise ValueError("unexpected active-count placement fields")
    elif kind == "shape_split":
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
    return dict(schema_version=1, deployment="independent_native_engines",
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
