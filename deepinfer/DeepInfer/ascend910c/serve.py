"""Start, inspect, or stop one owned 172 service using stable package paths."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROFILES = ("original-engine", "original-mask", "original-off", "original-deterministic", "native140", "official-yoco", "official-zos", "native256", "official-eplb-off", "official-eplb-on")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd(*args, timeout=35):
    return subprocess.check_output(args, text=True, timeout=timeout)


def stat(pid):
    return Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()


def birth(pid):
    return stat(pid)[19]


def live(pid, started):
    try:
        return stat(pid)[0] != "Z" and birth(pid) == str(started)
    except (FileNotFoundError, ProcessLookupError):
        return False


def signal_owned(pid, started, sig):
    try:
        if live(pid, started):
            os.kill(pid, sig)
    except ProcessLookupError:
        pass


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def plan(service, profile="original-engine", root=None):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]{0,63}", service):
        raise ValueError("invalid service name")
    if profile not in PROFILES:
        raise ValueError("unknown profile")
    config = read(HERE / "launch.json")
    legacy = profile in ("native140", "native256")
    root = Path(root).resolve() if root is not None else Path(config["legacy_source"]) if legacy else HERE.parent
    project = Path(config["project"])
    if not root.is_relative_to(project):
        raise ValueError("source must be inside the wio project")
    control = project / "serving" / service
    mapping = {"@SOURCE@": str(root), "@PROJECT@": str(project), "@SERVICE@": service}
    commands = {}
    for name, values in config["legacy_commands" if legacy else "commands"].items():
        commands[name] = values.copy()
        for index, value in enumerate(commands[name]):
            for key, replacement in mapping.items():
                value = value.replace(key, replacement)
            commands[name][index] = value
    mode = read(HERE / "profiles" / (profile + ".json"))
    if not legacy:
        settings = {
            "DEEPINFER_910C_SOFTWARE_OPS": "1", "DEEPINFER_SINGLE_POINT_ABLATION": "0",
            "DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE": mode.get("yoco_mode", "compact"),
            "DEEPINFER_YOCO_PREFILL_KEEP_TOKENS": str(mode.get("yoco_keep_tokens", 140)),
            "DEEPINFER_USE_NEXTN_DECODE": str(mode.get("nextn", 5)),
            "DEEPINFER_ENABLE_MTP_VARLEN": str(int(mode.get("mtp_varlen", True))),
            "DEEPINFER_ZERO_OVERHEAD_SCHEDULING": str(int(mode.get("zos", True))),
            "DEEPINFER_ENFORCE_EAGER": str(int(mode.get("enforce_eager", False))),
            "DEEPINFER_BATCH_INVARIANT": str(int(mode.get("batch_invariant", False))),
            "DEEPINFER_USE_MULTIMODAL_POOL": "0",
            "LD_PRELOAD": str(root / "ascend910c/lib/libstdc++.so.6"),
            "TMPDIR": str(project / "t" / service),
        }
        deps = config["software_dependencies"]
        for command in commands.values():
            for index, value in enumerate(command):
                if value.startswith("PYTHONPATH="):
                    paths = value.removeprefix("PYTHONPATH=").split(":")
                    command[index] = "PYTHONPATH=" + ":".join(dict.fromkeys([str(root), deps] + paths))
            for key, value in settings.items():
                positions = [i for i, item in enumerate(command) if item.startswith(key + "=")]
                if positions:
                    for index in positions:
                        command[index] = key + "=" + value
                else:
                    command[2:2] = ["-e", key + "=" + value]
        mode = dict(mode, yoco_mode=settings["DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE"],
                    yoco_keep_tokens=int(settings["DEEPINFER_YOCO_PREFILL_KEEP_TOKENS"]),
                    nextn=int(settings["DEEPINFER_USE_NEXTN_DECODE"]),
                    mtp_varlen=settings["DEEPINFER_ENABLE_MTP_VARLEN"] == "1",
                    zos=settings["DEEPINFER_ZERO_OVERHEAD_SCHEDULING"] == "1",
                    enforce_eager=settings["DEEPINFER_ENFORCE_EAGER"] == "1",
                    batch_invariant=settings["DEEPINFER_BATCH_INVARIANT"] == "1")
    if profile in ("official-eplb-off","official-eplb-on"):
        settings={"DEEPINFER_910C_EPLB":"1", "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS":"32",
                  "DEEPINFER_EXPERT_COUNT_WINDOW_SIZE":"3", "DEEPINFER_MOE_LB_STAT_INTERVAL":"128",
                  "DEEPINFER_REARRANGE_P2P_GROUPS":"1",
                  "DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL":"30" if profile.endswith("-on") else "0",
                  "TMPDIR":str(project/"t"/service)}
        worker=commands["worker_cmd"];at=worker.index("wio-deepinfer-s172-npu")
        worker[at:at]=[part for key,value in settings.items() for part in ("-e",key+"="+value)]
    manifest = HERE / "rollback-source-manifest.json" if legacy else root / "ascend910c/source-manifest.json"
    return dict(service=service, profile=profile, source=str(root), manager_source=str(HERE.parent), source_manifest=str(manifest), project=str(project),
                control=str(control), mode=mode, execution="native" if legacy else "original-engine",
                worker_command=commands["worker_cmd"], api_command=commands["api_cmd"])


def validate_ready(rows, spec):
    """Validate the actual Worker receipt, retaining frozen native recovery."""
    if sorted(row['rank'] for row in rows) != list(range(16)):
        raise RuntimeError('missing ready ranks')
    if spec.get('execution') != 'original-engine':
        if not all(row['dspark_graphs_initialized'] and row['target_vocab_parallel']
                   and row['mc2_capacity_floor'] == 4096 for row in rows):
            raise RuntimeError('native startup graphs differ')
        return
    expected = sha(Path(spec['source']) / 'ascend910c/source-manifest.json')
    mode = spec['mode']
    for row in rows:
        if (row.get('protocol') != 'deepinfer-original-engine-v1'
                or row.get('state') != 'READY' or row.get('source') != expected
                or row.get('world_size') != 16 or row.get('data_parallel_size') != 2
                or row.get('context_parallel_size') != 8 or row.get('dp_rank') != row['rank'] // 8
                or row.get('nextn') != mode['nextn']
                or row.get('capabilities', {}).get('batch_invariant', False) != mode.get('batch_invariant', False)
                or row.get('execution_options') != {key: mode[key] for key in
                    ('yoco_mode', 'yoco_keep_tokens', 'mtp_varlen', 'zos', 'enforce_eager')}):
            raise RuntimeError('original Engine ready configuration differs')
        sizes, curve = row.get('graph_sizes'), row.get('decode_perf_curve')
        if not isinstance(sizes, list) or not isinstance(curve, list):
            raise RuntimeError('original Engine graph receipt is incomplete')
        if mode['enforce_eager']:
            if sizes or curve:
                raise RuntimeError('eager receipt contains graph resources')
        elif not sizes or any(type(size) is not int or size < 1 for size in sizes):
            raise RuntimeError('original Engine graph sizes are invalid')
        if mode['mtp_varlen']:
            if (len(curve) != len(sizes) or [value[0] for value in curve] != sizes
                    or any(not math.isfinite(value[1]) or value[1] <= 0 for value in curve)):
                raise RuntimeError('original Engine decode costs are incomplete')
        elif curve:
            raise RuntimeError('fixed Engine receipt contains variable costs')


def verify_source(root):
    config = read(HERE / "launch.json")
    legacy = root == Path(config["legacy_source"])
    manifest = HERE / "rollback-source-manifest.json" if legacy else root / "ascend910c/source-manifest.json"
    for relative, expected in read(manifest).items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError("invalid source manifest path")
        path = (root / relative).resolve()
        if (not legacy and not path.is_relative_to(root)) or sha(path) != expected:
            raise RuntimeError("source identity differs: " + relative)
    return sha(manifest)


def containers():
    config = read(HERE / "launch.json")
    if socket.gethostname() != config["hostname"]:
        raise RuntimeError("unexpected server")
    result = {}
    alias = Path(config["model_alias"])
    for name, expected in config["containers"].items():
        value = json.loads(cmd("docker", "inspect", name))[0]
        if value["Id"] != expected or value["Image"] != config["image"] or not value["State"]["Running"]:
            raise RuntimeError("container identity differs: " + name)
        parents = [m for m in value["Mounts"] if alias.is_relative_to(Path(m["Destination"]))]
        mount = max(parents, key=lambda m: len(Path(m["Destination"]).parts))
        actual = (Path(mount["Source"]) / alias.relative_to(Path(mount["Destination"]))).resolve()
        if mount["RW"] or actual != Path(config["model_host"]).resolve():
            raise RuntimeError("model mount identity differs")
        for child in value["Mounts"]:
            destination = Path(child["Destination"])
            if destination == alias or destination.is_relative_to(alias):
                expected_child = Path(config["model_host"]) / destination.relative_to(alias)
                if child["RW"] or Path(child["Source"]).resolve() != expected_child.resolve():
                    raise RuntimeError("model child mount differs")
        result[name] = value["Id"]
    return result


def holders():
    table = cmd("npu-smi", "info")
    if "Process id" not in table:
        raise RuntimeError("unrecognized device table")
    return table, [(int(a), int(b), int(pid)) for a, b, pid in re.findall(
        r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|", table, re.M)]


def idle_devices():
    table, rows = holders()
    if rows or not all(f"No running processes found in NPU {i}" in table for i in range(8)):
        raise RuntimeError("devices occupied; no existing service is stopped by start")
    if any(cmd("ss", "-H", "-ltn", f"sport = :{port}").strip() for port in (8172, 8173)):
        raise RuntimeError("endpoint occupied")


def render(spec):
    config = read(HERE / "launch.json")
    body = (HERE / "controller-body.py.txt").read_text().replace("@SERVICE@", spec["service"])
    header = "from pathlib import Path\nimport subprocess,json,time,re,os,atexit,signal,socket,sys\n"
    header += "SPEC=json.loads(" + repr(json.dumps(spec)) + ")\n"
    header += "assert socket.gethostname()==" + repr(config["hostname"]) + "\n"
    header += "_service_lock=os.fdopen(int(sys.argv[1]))\n"
    header += "p=Path(SPEC['project']);control=Path(SPEC['control']);status=p/'runs'/('serving-'+SPEC['service']+'-status.json')\n"
    header += "assert not status.exists() and not control.exists(),'already dispatched'\n"
    header += "smi=subprocess.check_output(['npu-smi','info'],text=True)\n"
    header += "assert 'Process id' in smi and not re.search(r'^\\|\\s*\\d+\\s+\\d+\\s*\\|\\s*\\d+\\s*\\|',smi,re.M),'occupied'\n"
    header += "assert not subprocess.check_output(['ss','-H','-ltn','sport = :8172'],text=True).strip(),'endpoint occupied'\n"
    header += "control.mkdir(parents=True,exist_ok=False);(control/'single-point-mode.json').write_text(json.dumps(SPEC['mode']))\n"
    header += "worker_cmd=SPEC['worker_command'];api_cmd=SPEC['api_command']\n"
    header += "s=dict(state='LOADING',started=time.time(),worker_command=worker_cmd,api_command=api_cmd,source_manifest=SPEC['source_manifest'],profile=SPEC['profile'])\n"
    return header + body


def start(spec):
    project, root = Path(spec["project"]), Path(spec["source"])
    source_digest = verify_source(root)
    identities = containers()
    service = spec["service"]
    receipt = project / "runs" / ("serving-" + service + "-transition.json")
    controller = project / "runs" / ("serving-" + service + "-controller.py")
    log = project / "logs" / ("serving-" + service + "-controller.log")
    if any(p.exists() for p in (receipt, controller, log, Path(spec["control"]))):
        raise RuntimeError("service name already used; inspect its receipt")
    recipe = project / "dependencies/deepseek-recipe-0.2.0-20261029"
    manifest = recipe / "site-manifest.json"
    if sha(manifest) != read(HERE / "launch.json")["recipe_manifest_sha256"]:
        raise RuntimeError("tokenizer recipe manifest differs")
    for relative, digest in read(manifest)["files"].items():
        if sha(recipe / "site" / relative) != digest:
            raise RuntimeError("tokenizer recipe file differs")
    lock_path = project / "runs/910c-service.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle_devices()
        for path in [project / "cache" / (prefix + service) for prefix in ("home-", "xdg-", "work-", "triton-", "ascend-")]:
            path.mkdir(exist_ok=False)
        for path in (project / "tmp" / service, project / "logs" / ("ascend-" + service), project / "evidence" / service):
            path.mkdir(exist_ok=False)
        if spec.get('execution') == 'original-engine':
            if not Path(read(HERE / 'launch.json')['software_dependencies']).is_dir():
                raise RuntimeError('original framework dependencies are unavailable')
            (project/"t"/service).mkdir(parents=True,exist_ok=False)
        controller.write_text(render(spec))
        atomic(receipt, dict(state="SUBMITTING", started=time.time(), controller_script=str(controller)))
        idle_devices()
        with log.open("x") as output:
            process = subprocess.Popen([sys.executable, str(controller), str(lock.fileno())], stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        value = dict(state="SERVICE_DISPATCHED", pid=process.pid, starttime=birth(process.pid),
            dispatched=time.time(), log=str(log), containers=identities, source=str(root),
            source_manifest_sha256=source_digest, profile=spec["profile"], controller_script=str(controller),
            controller_sha256=sha(controller))
        atomic(receipt, value)
        return value


def inspect_owner(service):
    spec = plan(service)
    project, control = Path(spec["project"]), Path(spec["control"])
    identities = containers()
    receipt = read(project / "runs" / ("serving-" + service + "-transition.json"))
    status = read(project / "runs" / ("serving-" + service + "-status.json"))
    if receipt["state"] != "SERVICE_DISPATCHED" or status["state"] != "READY":
        raise RuntimeError("service not ready")
    root = Path(receipt["source"])
    if verify_source(root) != receipt["source_manifest_sha256"]:
        raise RuntimeError("source manifest differs")
    controller = receipt["pid"]
    if not live(controller, receipt["starttime"]) or sha(Path(receipt["controller_script"])) != receipt["controller_sha256"]:
        raise RuntimeError("controller identity differs")
    controller_args = Path("/proc", str(controller), "cmdline").read_bytes().split(b"\0")
    if receipt["controller_script"].encode() not in controller_args:
        raise RuntimeError("controller command differs")
    _, rows = holders()
    if len(rows) != 16 or {(a, b) for a, b, _ in rows} != {(a, b) for a in range(8) for b in (0, 1)}:
        raise RuntimeError("unexpected device geometry")
    npu = identities["wio-deepinfer-s172-npu"]
    owned = {int(line.strip()) for line in cmd("docker", "top", npu, "-eo", "pid").splitlines()[1:] if line.strip().isdigit()}
    devices = []
    for a, b, pid in rows:
        proc = Path("/proc", str(pid));argv = proc.joinpath("cmdline").read_bytes().split(b"\0")
        env = dict(item.split("=", 1) for item in proc.joinpath("environ").read_bytes().decode().split("\0") if "=" in item)
        if pid not in owned or str(control).encode() not in argv or b"examples/ascend910c_w8a8_worker.py" not in argv:
            raise RuntimeError("foreign device owner")
        if env["PYTHONPATH"].split(":")[0] != str(root):
            raise RuntimeError("worker source differs")
        rank = int(env["LOCAL_RANK"])
        if sorted(os.sched_getaffinity(pid)) != list(range(rank * 40 + 2, rank * 40 + 38)):
            raise RuntimeError("worker affinity differs")
        devices.append(dict(device=[a, b], pid=pid, starttime_ticks=birth(pid), rank=rank))
    if sorted(row["rank"] for row in devices) != list(range(16)):
        raise RuntimeError("worker ranks differ")
    spec = plan(service, receipt['profile'], root)
    ready = [read(path) for path in control.glob("ready-rank*.json")]
    # Frozen pre-integration services retain their native receipt and health.
    if ready and ready[0].get('protocol') != 'deepinfer-original-engine-v1':
        spec['execution'] = 'native'
    runtime = read(root / "ascend910c/runtime.json")["extension"] if spec['execution'] == 'native' and receipt["profile"] not in ("native140", "native256") else None
    validate_ready(ready, spec)
    for row in ready:
        if runtime is not None:
            identity = row["official_metadata_binding"]
            if identity["path"] != str(root / runtime["path"]) or identity["sha256"] != runtime["sha256"]:
                raise RuntimeError("worker extension differs")
    candidates = []
    for line in cmd("docker", "top", identities["wio-deepinfer-s172"], "-eo", "pid,args").splitlines()[1:]:
        values = line.split(None, 1)
        if len(values) == 2 and str(control) in values[1] and "ascend910c_w8a8_api_fixed_routing.py" in values[1]:
            candidates.append(int(values[0]))
    if len(candidates) != 1:
        raise RuntimeError("frontend identity ambiguous")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:8172/health", timeout=5) as response:
        health = json.load(response)
    if health != dict(status="ok", backend="DeepInfer-original" if spec['execution'] == 'original-engine' else "DeepInfer-native"):
        raise RuntimeError("health differs")
    return dict(state="SERVICE_OWNERS_AND_HEALTH_OBSERVED", client_id=identities["wio-deepinfer-s172"],
        worker_container_id=npu, controller_pid=controller, controller_starttime=receipt["starttime"],
        api_pid=candidates[0], api_starttime=birth(candidates[0]), devices=devices, health=health)


def completed(control):
    original = (control / 'ready-rank0.json').exists() and read(control / 'ready-rank0.json').get('protocol') == 'deepinfer-original-engine-v1'
    required = {"request.json", "result-dp0.json", "result-dp1.json"}
    if not original:
        required |= {f"hash-proof-rank{r}.json" for r in range(16)} | {f"ablation-rank{r}.json" for r in range(16)}
    for job in (control / "jobs").iterdir():
        if list(job.glob("error-rank*.txt")) or not required <= {p.name for p in job.iterdir()}:
            raise RuntimeError("unfinished or failed job: " + job.name)
        if any((read(job / f"result-dp{dp}.json").get('protocol') != 'deepinfer-original-engine-v1'
                if original else read(job / f"result-dp{dp}.json").get('state') != 'PASS') for dp in (0, 1)):
            raise RuntimeError("incomplete job result")


def stop(service):
    owner = inspect_owner(service)
    spec = plan(service);control = Path(spec["control"]);project = Path(spec["project"])
    completed(control)
    if cmd("ss", "-H", "-tn", "state", "established", "sport = :8172").strip():
        raise RuntimeError("active client; service retained")
    tracked = {owner["controller_pid"]: str(owner["controller_starttime"])}
    for container in (owner["client_id"], owner["worker_container_id"]):
        for line in cmd("docker", "top", container, "-eo", "pid,args").splitlines()[1:]:
            values = line.split(None, 1)
            if len(values) == 2 and str(control) in values[1]:
                if not any(text in values[1] for text in ("ascend910c_w8a8_worker.py", "ascend910c_w8a8_api_fixed_routing.py", "torch.distributed.run")):
                    raise RuntimeError("unexpected owned process")
                pid = int(values[0]);tracked[pid] = birth(pid)
    status = read(project / "runs" / ("serving-" + service + "-status.json"))
    for key in ("worker_host_pid", "api_host_pid"):
        pid = status.get(key)
        if pid and Path("/proc", str(pid)).exists():
            if str(control).encode() not in Path("/proc", str(pid), "cmdline").read_bytes():
                raise RuntimeError("wrapper identity differs")
            tracked[pid] = birth(pid)
    if any(pid not in tracked for _, _, pid in holders()[1]):
        raise RuntimeError("foreign device owner")
    api = owner["api_pid"];frozen = False;stopping = False
    try:
        if not live(api, owner["api_starttime"]):
            raise RuntimeError("frontend changed")
        signal_owned(api, owner["api_starttime"], signal.SIGSTOP);frozen = True
        if cmd("ss", "-H", "-tn", "state", "established", "sport = :8172").strip():
            raise RuntimeError("new client; service retained")
        completed(control)
        stopping = True
        for pid, started in tracked.items():
            if live(pid, started):
                signal_owned(pid, started, signal.SIGTERM)
                if pid == api and live(pid, started):
                    signal_owned(pid, started, signal.SIGCONT)
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline:
            if not any(live(pid, started) for pid, started in tracked.items()):
                value = dict(state="SERVICE_STOPPED", service=service, observed=time.time(), owned_pids=list(tracked))
                atomic(project / "runs" / ("serving-" + service + "-stopped.json"), value)
                return value
            time.sleep(1)
        raise TimeoutError("owned processes still exiting; inspect same receipt")
    finally:
        if frozen and not stopping and live(api, owner["api_starttime"]):
            signal_owned(api, owner["api_starttime"], signal.SIGCONT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "start", "inspect", "stop"))
    parser.add_argument("--service", required=True)
    parser.add_argument("--profile", choices=PROFILES, default="original-engine")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    spec = plan(args.service, args.profile, args.source_root)
    if args.action == "plan":
        result = spec
    elif args.source_root is not None:
        parser.error("source-root is a plan-only option")
    elif args.action == "start":
        result = start(spec)
    elif args.action == "inspect":
        result = inspect_owner(args.service)
    else:
        result = stop(args.service)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
