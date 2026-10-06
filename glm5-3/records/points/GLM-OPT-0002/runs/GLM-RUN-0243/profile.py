"""One existing-controller dynamic profile; no native source/service changes."""
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.request
R = Path(__file__).parent
G = R.parents[4]
sys.path.insert(0, str(G / "runtime"))
from phase_runner import atomic_json, process_identity, same_process, utc
from owner_guard import start_watchdog, guard
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


def clock(label):
    a = time.monotonic_ns()
    wall = time.time_ns()
    b = time.monotonic_ns()
    atomic_json(R / (label + "_clock.json"), dict(monotonic_before_ns=a, realtime_ns=wall,
                                               monotonic_after_ns=b))


def main():
    start_watchdog()
    before = {"D0": probe("166", d0["roots"]["node0"]), "D1": probe("167", d1["roots"]["node0"])}
    assert same_process(public_owner["identity"])
    idle("166", 9081)
    idle("167", 9900)
    store = d1_store("store_before")
    ranks = []
    for pid in before["D0"]["npu_worker_pids"]:
        stat = Path(f"/proc/{pid}/status").read_text()
        ns = re.search(r"^NSpid:\s+(.*)$", stat, re.M).group(1).split()
        title = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        match = re.search(r"PP(\d+)_TP(\d+)", title)
        assert match, title
        ranks.append(dict(host_pid=pid, container_pid=int(ns[-1]), pp=int(match[1]), tp=int(match[2]),
                          identity=process_identity(pid), title=title.strip()))
        env = dict(x.decode().split("=", 1) for x in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in x)
        assert env["PROFILING_MODE"] == "dynamic"
    assert {(x["pp"], x["tp"]) for x in ranks} == {(pp, tp) for pp in range(2) for tp in range(8)}
    atomic_json(R / "preflight.json", dict(at=utc(), domains=before, ranks=ranks, public=public_owner))
    cli = "/usr/local/Ascend/ascend-toolkit/latest/bin/msprof"
    args = [cli, "--dynamic=on", "--pid=" + ",".join(str(x["container_pid"]) for x in ranks),
            "--duration=15", "--output=" + str(R / "device_trace"), "--ascendcl=on", "--runtime-api=on",
            "--task-time=l1", "--ai-core=on", "--parse=off", "--export=off", "--analyze=off"]
    clock("before")
    shell = "source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; exec " + shlex.join(args)
    argv = ["docker", "exec", "glm52-single", "bash", "-c", shell]
    with (R / "msprof.stdout").open("xb") as out, (R / "msprof.stderr").open("xb") as err:
        prof = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
        atomic_json(R / "profiler_owner.json", dict(identity=process_identity(prof.pid), argv=argv,
                                                   target_ranks=ranks, duration_s=15))
        try:
            # Final device/rank/time coverage, not process liveness, accepts the
            # profile; delay only prevents generation after immediate CLI error.
            for _ in range(6):
                guard()
                time.sleep(.5)
                assert prof.poll() is None, "profiler exited before load; no generation"
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                clients = list(pool.map(client, range(4)))
            atomic_json(R / "client_summary.json", dict(at=utc(), rows=clients, complete=4, effective_output_tokens=256))
        finally:
            code = prof.wait(timeout=120)
            atomic_json(R / "profiler_exit.json", dict(at=utc(), exit_code=code))
    clock("after")
    assert code == 0, "profiler command failed; preserve raw"
    artifacts = [dict(path=str(p), bytes=p.stat().st_size) for p in (R / "device_trace").rglob("*") if p.is_file()]
    atomic_json(R / "device_inventory.json", dict(files=artifacts))
    assert any(x["bytes"] > 0 for x in artifacts), "no device data; no blind retry"
    idle("166", 9081)
    idle("167", 9900)
    after = {"D0": probe("166", d0["roots"]["node0"]), "D1": probe("167", d1["roots"]["node0"])}
    assert all(before[k]["root"] == after[k]["root"] and before[k]["npu_worker_pids"] == after[k]["npu_worker_pids"] for k in before)
    assert same_process(public_owner["identity"])
    assert d1_store("store_after") == store
    atomic_json(R / "final_site.json", dict(at=utc(), native_domains=after,
                 native_epochs_unchanged=True, D1_STORE_unchanged=True, service_operations=0,
                 public_health=json.loads(request(8000, "/healthcheck")), Current=None))


if __name__ == "__main__":
    main()
