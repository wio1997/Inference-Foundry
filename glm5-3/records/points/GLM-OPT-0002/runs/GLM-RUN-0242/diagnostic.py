"""One trace-only native window; existing controller owns all site operations."""
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

R = Path(__file__).parent
G = R.parents[4]
sys.path.insert(0, str(G / "runtime"))
from phase_runner import atomic_json, process_identity, same_process, utc
from owner_guard import start_watchdog, guard

SITE = Path("/data/tiankuan/wio/glm52-pd/deploy")
BUNDLE = R.parent / "GLM-RUN-0241/runtime_bundle"
TRACE_SOURCE = G / "records/points/GLM-OPT-0002/research/engine_commit_20261006"
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
material = json.loads((R / "initial_material.json").read_text())
d0 = next(e for e in material["engines"] if e["replica_id"] == "D0")
d1 = next(e for e in material["engines"] if e["replica_id"] == "D1")
public_owner = json.loads((R / "initial_public_owner.json").read_text())
counter = 0


def run(node, args, label, input=None, timeout=90):
    global counter
    guard()
    counter += 1
    if node == "167":
        args = ["ssh", "-o", "BatchMode=yes", "root@172.16.10.167", shlex.join(args)]
    p = subprocess.run(args, input=input, capture_output=True, timeout=timeout)
    stem = R / f"{counter:04d}_{label}"
    stem.with_suffix(".stdout").write_bytes(p.stdout)
    stem.with_suffix(".stderr").write_bytes(p.stderr)
    atomic_json(stem.with_suffix(".receipt.json"), dict(argv=args, exit_code=p.returncode,
                at=utc(), stdout_sha256=hashlib.sha256(p.stdout).hexdigest()))
    p.check_returncode()
    return p.stdout


def request(port, path, body=None, method=None, timeout=30):
    guard()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method=method)
    with HTTP.open(req, timeout=timeout) as response:
        return response.read()


def idle(node, port):
    with HTTP.open(f"http://172.16.10.{node}:{port}/metrics", timeout=10) as response:
        raw = response.read()
    values = re.findall(r"^vllm:num_requests_(?:running|waiting)\{[^\n]*\}\s+([0-9.eE+-]+)", raw.decode(), re.M)
    assert values and all(float(x) == 0 for x in values)
    return raw


def probe(node, root, count=16):
    return json.loads(run(node, ["python3", "-c", (R / "live_probe.py").read_text()],
                         "probe" + node, json.dumps(dict(owner=copy.deepcopy(root), NPU_count=count)).encode()))


def stop(root, members, preflight=False):
    return run("166", ["python3", "-c", (R / "stop_prior_cohort.py").read_text()], "ownedstop",
        json.dumps(dict(owner=root, members=members["owned_targets"], preflight=preflight)).encode(), timeout=120)


def start_native(plan, label, traced=False):
    argv = plan["argv"]
    plugin = Path(argv[1]).parent
    env = dict(plan["environment"])
    if traced:
        env["GLM_COMMIT_TRACE_DIR"] = str(R / "host_trace")
    shell = "ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source " + shlex.quote(str(SITE / "scripts/pd_common_env.sh"))
    shell += "; export PYTHONPATH=" + shlex.quote(str(TRACE_SOURCE) + ":" + str(plugin) if traced else str(plugin)) + ":$PYTHONPATH VLLM_HOST_IP=172.16.10.166 VLLM_ENABLE_RESPONSES_API_STORE=1 "
    shell += " ".join(k + "=" + shlex.quote(v) for k, v in env.items())
    shell += "; exec " + shlex.join(argv) + " > " + shlex.quote(plan["log"]) + " 2>&1"
    assert not Path(plan["log"]).exists()
    run("166", ["docker", "exec", "-d", "glm52-single", "bash", "-c", shell], label + "start")
    root = dict(host="166", argv=argv, role="API")
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        guard()
        try:
            live = probe("166", root, None)
        except subprocess.CalledProcessError:
            if "pid" in root:
                raise
            time.sleep(2)
            continue
        root = live["root"]
        atomic_json(R / (label + "_root.json"), root)
        atomic_json(R / (label + "_startup_members.json"), live)
        text = Path(plan["log"]).read_text(errors="replace")[-18000:]
        if any(t in text for t in ("Engine core initialization failed", "Traceback (most recent call last):", "OutOfMemoryError")):
            raise RuntimeError(label + " startup failed; see immutable native log")
        try:
            request(9081, "/health", timeout=3)
            live = probe("166", root)
            atomic_json(R / (label + "_members.json"), live)
            return root, live
        except (OSError, AssertionError):
            pass
        time.sleep(10)
    raise TimeoutError(label + " startup timeout")


def retire_public(owner):
    identity = owner["identity"]
    assert same_process(identity)
    actual = [x.decode() for x in Path(f"/proc/{identity['pid']}/cmdline").read_bytes().split(b"\0") if x]
    assert actual == owner["argv"]
    assert json.loads(request(8000, "/healthcheck"))["request_num"] == 0
    assert all(not r["active_requests"] for r in json.loads(request(8000, "/control/replicas"))["replicas"])
    guard()
    os.kill(identity["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 45
    while same_process(identity) and time.monotonic() < deadline:
        guard()
        time.sleep(.2)
    assert not same_process(identity)
    log = Path(owner["argv"][owner["argv"].index("--config") + 1]).parent / "public.gateway.log"
    acks = [json.loads(l) for l in log.read_text().splitlines() if l.startswith('{"event":')]
    assert [x["event"] for x in acks] == ["task_acl_init", "task_acl_finalize"]
    assert all(x["returncode"] == 0 for x in acks)


def start_public(restored_material):
    folder = R / "restored"
    folder.mkdir()
    atomic_json(folder / "native_engines_resident.json", restored_material)
    sys.path.insert(0, str(BUNDLE))
    from native_engines_service_config import render, checked_config
    config = render(folder, R.parent / "GLM-RUN-0125")
    atomic_json(folder / "service_config.json", config)
    checked_config(folder / "service_config.json")
    retire_public(public_owner)
    argv = ["/usr/local/python3.12.13/bin/python3", str(SITE / "plugins/local_engines137/native_acl_lifecycle.py"),
            "script", str(BUNDLE / "native_engines_service_entry_v14.py"), "--config", str(folder / "service_config.json"),
            "--host", "0.0.0.0", "--port", "8000"]
    log = folder / "public.gateway.log"
    shell = "ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source " + shlex.quote(str(SITE / "scripts/pd_common_env.sh"))
    shell += "; export PYTHONPATH=" + shlex.quote(str(BUNDLE)) + ":$PYTHONPATH; exec " + shlex.join(argv) + " > " + shlex.quote(str(log)) + " 2>&1"
    run("166", ["docker", "exec", "-d", "glm52-single", "bash", "-c", shell], "publicstart")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            if json.loads(request(8000, "/healthcheck"))["status"] == "ok":
                break
        except OSError:
            pass
        time.sleep(1)
    else:
        raise TimeoutError("public readiness")
    text = log.read_text()
    marker = next(json.loads(l.split("GLM_SERVICE_ENTRY_INSTALLED ")[1]) for l in text.splitlines() if "GLM_SERVICE_ENTRY_INSTALLED " in l)
    top = subprocess.check_output(["docker", "top", "glm52-single", "-eo", "pid,args"], text=True)
    ids = [int(l.split()[0]) for l in top.splitlines()[1:] if str(folder / "service_config.json") in l]
    assert len(ids) == 1
    actual = [v.decode() for v in Path(f"/proc/{ids[0]}/cmdline").read_bytes().split(b"\0") if v]
    assert actual == argv
    owner = dict(identity=process_identity(ids[0]), argv=argv)
    atomic_json(folder / "public_host_owner.json", owner)
    obsdir = folder / "identity_observer"
    obsdir.mkdir()
    obsargv = ["/usr/bin/python3", str(BUNDLE / "native_identity_observer.py"), "--config", str(folder / "service_config.json"),
               "--public-owner", str(folder / "public_host_owner.json"), "--output-dir", str(obsdir), "--interval-s", "5"]
    with (obsdir / "process.log").open("xb") as out:
        observer = subprocess.Popen(obsargv, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(obsdir / "process_owner.json", dict(identity=process_identity(observer.pid), argv=obsargv))
    for _ in range(60):
        guard()
        assert observer.poll() is None
        if (obsdir / "latest.json").exists() and all(x["status"] == "healthy" for x in json.loads((obsdir / "latest.json").read_text())["groups"]):
            break
        time.sleep(.5)
    else:
        raise RuntimeError("fresh identity observer did not confirm both groups")
    atomic_json(folder / "public_summary.json", dict(owner=owner, marker=marker, health=json.loads(request(8000, "/healthcheck"))))


def client(index):
    body = dict(model="glm-52", messages=[dict(role="user", content=f"List numbered short facts about the Moon. Request {index}.")],
                temperature=0, seed=1024, max_tokens=64, ignore_eos=True, stream=True,
                stream_options=dict(include_usage=True), return_token_ids=True,
                chat_template_kwargs=dict(enable_thinking=False))
    data = json.dumps(body).encode()
    (R / f"request{index}.body").write_bytes(data)
    req = urllib.request.Request("http://127.0.0.1:9081/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
    events, wire = [], bytearray()
    began = time.monotonic_ns()
    with HTTP.open(req, timeout=180) as response:
        for line in response:
            now = time.monotonic_ns()
            wire.extend(line)
            if line.startswith(b"data: "):
                raw = line[6:].strip()
                if raw == b"[DONE]":
                    events.append(dict(monotonic_ns=now, done=True))
                else:
                    events.append(dict(monotonic_ns=now, payload=json.loads(raw)))
    (R / f"request{index}.wire").write_bytes(wire)
    atomic_json(R / f"request{index}.events.json", dict(begin_monotonic_ns=began, events=events))
    usage = [x["payload"]["usage"] for x in events if "payload" in x and x["payload"].get("usage")]
    ids = {x["payload"]["id"] for x in events if "payload" in x}
    assert events[-1].get("done") and len(ids) == 1 and usage[-1]["completion_tokens"] == 64
    return dict(index=index, id=ids.pop(), usage=usage[-1], done=True)


def d1_store(label):
    rows = {}
    for response_id in ("resp_glm_run241_base", "resp_glm_run241_bg32"):
        with HTTP.open("http://172.16.10.167:9900/v1/responses/" + response_id, timeout=10) as response:
            raw = response.read()
        (R / (label + "_" + response_id + ".wire")).write_bytes(raw)
        value = json.loads(raw)
        assert value["id"] == response_id and value["status"] in ("completed", "incomplete")
        rows[response_id] = value
    return rows


def main():
    start_watchdog()
    identity = json.loads((TRACE_SOURCE / "source_identity.json").read_text())
    check = "import json,hashlib,sys;from pathlib import Path;s=json.load(sys.stdin)['sources'];a=[dict(path=x['path'],sha256=hashlib.sha256(Path(x['path']).read_bytes()).hexdigest())for x in s];assert all(x['sha256']==y['sha256']for x,y in zip(s,a));print(json.dumps(a))"
    run("166", ["docker", "exec", "-i", "glm52-single", "python3", "-c", check],
        "installed_sources", json.dumps(identity).encode())
    oldroot = d0["roots"]["node0"]
    oldmembers = probe("166", oldroot)
    d1members = probe("167", d1["roots"]["node0"])
    assert oldmembers["npu_worker_pids"] == d0["members"]["node0"]["npu_worker_pids"]
    idle("166", 9081)
    idle("167", 9900)
    stop(oldroot, oldmembers, True)
    store_before = d1_store("store_before")
    atomic_json(R / "preflight.json", dict(at=utc(), D0=oldmembers, D1=d1members,
               health=json.loads(request(8000, "/healthcheck"))))
    config = json.loads(Path(public_owner["argv"][public_owner["argv"].index("--config") + 1]).read_text())
    domain = next(g for g in config["native_domains"] if "D0" in g["members"])
    request(8000, "/control/native-groups/" + domain["id"] + "/fault",
            dict(epoch=domain["epoch"], reason="Run242 trace-only epoch; old native STORE retires, no migration"))
    stop(oldroot, oldmembers)
    original = copy.deepcopy(d0["plans"]["node0"])
    traced = copy.deepcopy(original)
    for flag, value in (("--worker-cls", "commit_trace_worker.TraceWorker"),
                        ("--scheduler-cls", "commit_trace_scheduler.BudgetScheduler")):
        traced["argv"][traced["argv"].index(flag) + 1] = value
    traced["argv"] += ["--profiler-config", json.dumps(dict(profiler="torch", torch_profiler_dir=str(R / "device_trace"),
                         torch_profiler_with_stack=False, torch_profiler_with_memory=False, ignore_frontend=True))]
    traced["log"] = str(R / "diagnostic_native.log")
    trace_root = None
    error = None
    try:
        trace_root, trace_members = start_native(traced, "trace", True)
        request(9081, "/start_profile", method="POST", timeout=90)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(client, range(4)))
            atomic_json(R / "client_summary.json", dict(at=utc(), rows=rows, complete=4, effective_output_tokens=256))
        finally:
            request(9081, "/stop_profile", method="POST", timeout=300)
        idle("166", 9081)
    except BaseException as exc:
        error = repr(exc)
        atomic_json(R / "diagnostic_error.json", dict(at=utc(), error=error))
    finally:
        # Restore native semantics even when startup/profiler/client fails.
        if trace_root is None and (R / "trace_root.json").exists():
            trace_root = json.loads((R / "trace_root.json").read_text())
        if trace_root is not None:
            try:
                trace_members = probe("166", trace_root, None)
            except subprocess.CalledProcessError:
                # Never broaden signals: original identities are rechecked by
                # the existing stop helper; unknown live resources stop cleanup.
                trace_members = json.loads((R / "trace_startup_members.json").read_text())
            stop(trace_root, trace_members)
        original["log"] = str(R / "restored_native.log")
        restored_root, restored_members = start_native(original, "restored")
        d0.update(id="local-242R-166", roots=dict(node0=restored_root),
                  members=dict(node0=restored_members), plans=dict(node0=original))
        d1["members"]["node0"] = probe("167", d1["roots"]["node0"])
        idle("166", 9081)
        idle("167", 9900)
        start_public(material)
        assert d1_store("store_after") == store_before
        atomic_json(R / "restoration.json", dict(at=utc(), original_native_code=True, V14_public=True,
                    D1_epoch_unchanged=True, workers=32, old_D0_STORE_retired=True, Current=None,
                    diagnostic_error=error, status="restored"))
    if error:
        raise RuntimeError("diagnostic failed, original services restored: " + error)


if __name__ == "__main__":
    main()
