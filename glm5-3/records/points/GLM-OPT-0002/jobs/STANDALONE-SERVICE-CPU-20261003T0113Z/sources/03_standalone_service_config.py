"""Compile a standalone native engine's API and headless process domain.

No live process checks or native operations happen here. The controller must
verify every root and NPU descendant before exposing this recorded domain.
"""
import argparse
import hashlib
import json
from pathlib import Path

EVIDENCE_NAMES = ("standalone_launch.json", "standalone_root_identities.json",
                  "standalone_native_members.json")


def _integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def _identity(identity):
    if set(identity) != {"boot_id", "start_ticks"}:
        raise ValueError("process identity requires boot_id and start_ticks")
    if not isinstance(identity["boot_id"], str) or not identity["boot_id"]:
        raise ValueError("missing boot identity")
    if not str(identity["start_ticks"]).isdigit() or int(identity["start_ticks"]) <= 0:
        raise ValueError("invalid start identity")


def _option(argv, name):
    if argv.count(name) != 1:
        raise ValueError("one explicit native CLI option required: " + name)
    value = argv[argv.index(name) + 1]
    if not isinstance(value, str):
        raise ValueError("native CLI values must be strings")
    return value


def _geometry(argv):
    options = {"TP": "--tensor-parallel-size",
               "DCP": "--decode-context-parallel-size",
               "PP": "--pipeline-parallel-size",
               "PCP": "--prefill-context-parallel-size",
               "DP": "--data-parallel-size", "nnodes": "--nnodes"}
    geometry = {key: _integer(int(_option(argv, option)), key)
                for key, option in options.items()}
    geometry["world_size"] = geometry["TP"] * geometry["PP"] * geometry["PCP"]
    if geometry["DP"] != 1 or geometry["nnodes"] not in (1, 2):
        raise ValueError("standalone service supports one DP1 engine on one or two nodes")
    if geometry["world_size"] % geometry["nnodes"]:
        raise ValueError("native world size must divide evenly across nodes")
    geometry["local_world_size"] = geometry["world_size"] // geometry["nnodes"]
    if geometry["local_world_size"] > 16:
        raise ValueError("recorded engine exceeds local sixteen-NPU allocation")
    if "--kv-transfer-config" in argv:
        raise ValueError("standalone service must use native local prefill without PD")
    if _option(argv, "--distributed-executor-backend") != "mp":
        raise ValueError("recorded roots must use the native multiproc executor")
    return geometry


def compile_config(plans, roots, members, state_dir):
    state_dir = Path(state_dir)
    if not state_dir.is_absolute():
        raise ValueError("service state directory must be absolute")
    if not isinstance(plans, dict) or not plans or set(plans) != set(roots) or set(plans) != set(members):
        raise ValueError("each native root requires launch, identity and membership evidence")
    geometry = None
    base_argv = None
    nodes = set()
    ranks = set()
    physical = set()
    api_root = None
    recorded = []
    for key in sorted(plans):
        plan, root, member = plans[key], roots[key], members[key]
        host = plan["host"]
        if host not in ("166", "167") or host in nodes:
            raise ValueError("one recorded native root per task host is required")
        nodes.add(host)
        argv = plan["argv"]
        if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            raise ValueError("recorded CLI must be a list of strings")
        if root["host"] != host or root["argv"] != argv or member["root"] != root or member["root_alive"] is not True:
            raise ValueError("native root must match its launch and member snapshots")
        _integer(root["pid"], "root PID")
        _identity(root["identity"])
        current = _geometry(argv)
        rank = int(_option(argv, "--node-rank"))
        if rank != plan["node_rank"] or rank < 0 or rank >= current["nnodes"] or rank in ranks:
            raise ValueError("invalid or duplicate native node rank")
        ranks.add(rank)
        headless = "--headless" in argv
        if headless != (rank != 0) or plan["api"] is not (rank == 0):
            raise ValueError("native node0 exposes API; remaining nodes must be headless")
        if root["role"] != ("API" if rank == 0 else "headless"):
            raise ValueError("root role disagrees with native CLI")
        normalized = argv[:]
        index = normalized.index("--node-rank")
        del normalized[index:index + 2]
        if headless:
            normalized.remove("--headless")
        if geometry is None:
            geometry, base_argv = current, normalized
        elif geometry != current or base_argv != normalized:
            raise ValueError("nodes must have the same native model and engine arguments")
        if current["nnodes"] > 1:
            if _option(argv, "--master-addr") != "172.16.10.166":
                raise ValueError("coupled engine requires its recorded166 master address")
            _integer(int(_option(argv, "--master-port")), "master port")
            if rank == 0 and host != "166":
                raise ValueError("native coupled node0 must belong to166")
        workers = member["npu_worker_pids"]
        if not isinstance(workers, list) or len(workers) != current["local_world_size"] or len(set(workers)) != len(workers):
            raise ValueError("complete local NPU membership is required")
        targets = {}
        for row in member["owned_targets"]:
            pid = _integer(row["pid"], "descendant PID")
            if pid in targets:
                raise ValueError("duplicate recorded descendant")
            targets[pid] = row
            _identity(row["identity"])
            if row["identity"]["boot_id"] != root["identity"]["boot_id"]:
                raise ValueError("root and descendants must belong to the same host boot")
        native = []
        for pid in workers:
            _integer(pid, "NPU worker PID")
            if pid == root["pid"] or pid not in targets:
                raise ValueError("NPU worker must belong to root ancestry")
            identity = targets[pid]["identity"]
            pair = (host, pid)
            if pair in physical:
                raise ValueError("native NPU workers must not overlap")
            physical.add(pair)
            native.append(dict(pid=pid, identity=identity))
        recorded.append(dict(key=key, launch=plan, root=root,
                             native_workers=sorted(native, key=lambda x: x["pid"])))
        if rank == 0:
            port = _integer(int(_option(argv, "--port")), "API port")
            if port > 65535:
                raise ValueError("invalid API port")
            api_root = dict(host=host, port=port)
    if len(plans) != geometry["nnodes"] or ranks != set(range(geometry["nnodes"])):
        raise ValueError("all native engine nodes must belong to the shared domain")
    if len(physical) != geometry["world_size"] or api_root is None:
        raise ValueError("complete native engine and API domain are required")
    epoch = hashlib.sha256(json.dumps(dict(geometry=geometry, roots=recorded),
                                      sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    replicas = [dict(id="D0", url="http://172.16.10." + api_root["host"] + ":" + str(api_root["port"]))]
    groups = [dict(id="standalone-native", epoch=epoch, members=["D0"])]
    environment = {
        "GLM_REPLICAS": json.dumps(replicas),
        "GLM_EXECUTION_GROUPS": json.dumps(groups),
        "GLM_PLACEMENT_POLICY": "active_count",
        "GLM_PD_PRODUCERS": "{}",
        "GLM_PD_NATIVE_PLANS": "{}",
        "GLM_RESPONSE_OWNER_STATE_PATH": str(state_dir / "response_owners.json"),
        "GLM_ROUTER_TRACE_PATH": str(state_dir / "router_trace.jsonl"),
        "GLM_ROUTER_AUDIT_DIR": str(state_dir / "router_wire"),
        "GLM_PD_AUDIT_DIR": str(state_dir / "native_PD_raw"),
    }
    return dict(schema_version=1, deployment="standalone_native", geometry=geometry,
                environment=environment, native_domains=groups,
                limitations=[
                    "Recorded roots/NPU ancestry require fresh controller live checks",
                    "All coupled nodes form one native engine and Responses store fault domain",
                    "Native process replacement retires this epoch; STORE is not replicated",
                    "No PD helper or native KV transfer is configured",
                    "Recorded CLI and members do not certify GPU correctness or capacity",
                ])


def render(resident_dir, state_dir):
    material = [json.loads((resident_dir / name).read_text()) for name in EVIDENCE_NAMES]
    config = compile_config(*material, state_dir)
    config["resident_evidence"] = {
        name: dict(path=str(resident_dir / name), sha256=hashlib.sha256(
            (resident_dir / name).read_bytes()).hexdigest()) for name in EVIDENCE_NAMES}
    return config


def checked_config(path):
    config = json.loads(path.read_text())
    evidence = config["resident_evidence"]
    if set(evidence) != set(EVIDENCE_NAMES):
        raise ValueError("complete standalone resident evidence is required")
    material = []
    for name in EVIDENCE_NAMES:
        item = evidence[name]
        raw = Path(item["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("recorded standalone evidence changed")
        material.append(json.loads(raw))
    state_dir = Path(config["environment"]["GLM_RESPONSE_OWNER_STATE_PATH"]).parent
    expected = compile_config(*material, state_dir)
    if set(config) != set(expected) | {"resident_evidence"} or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("standalone config does not match its recorded native domain")
    return config, state_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resident-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(render(args.resident_dir, args.state_dir), indent=2) + "\n")


if __name__ == "__main__":
    main()
