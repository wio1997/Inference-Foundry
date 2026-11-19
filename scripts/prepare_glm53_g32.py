"""从固定 SGLang 包重建 G32，并按 G316 历史配置生成独立 P/D 启动目录。"""
import argparse
import hashlib
import shlex
import ipaddress
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "environment/glm53_g32_source.json"
SPEC = json.loads(SPEC_PATH.read_text())
G316 = ROOT / "code/sg_pd_g316"
INDEX = "quant_model_weights.safetensors.index.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def check_source(root, expected, source_checkout=False):
    files = {
        f"sglang/{p.relative_to(root).as_posix()}": sha(p)
        for p in root.rglob("*")
        if "__pycache__" not in p.parts and p.is_file()
    }
    for path, digest in SPEC["image_only_unchanged_files"].items():
        recorded = files.pop(path, None)
        if recorded != digest and not (source_checkout and recorded is None):
            raise ValueError(f"镜像附带文件与记录不符：{path}")
    listing = "".join(f"{digest}  {path}\n" for path, digest in sorted(files.items()))
    if (len(files) != SPEC["base_source_file_count"]
            or hashlib.sha256(listing.encode()).hexdigest() != SPEC[expected]):
        changed = SPEC["changed_files"]
        key = "before_sha256" if expected == "base_source_tree_sha256" else "after_sha256"
        wrong = [path for path, pin in changed.items() if files.get(path) != pin[key]]
        raise ValueError(f"SGLang 源树校验失败：{expected}，补丁目标差异：{wrong}")


def checkpoint_receipt(model, identity_path):
    identity = json.loads(identity_path.read_text())
    index = model / INDEX
    if sha(index) != identity["index_sha256"]:
        raise ValueError("checkpoint 索引与历史权重不符")
    weights = json.loads(index.read_text())["weight_map"]
    for shard in set(weights.values()):
        if Path(shard).name != shard or not (model / shard).is_file():
            raise ValueError(f"checkpoint 分片缺失或路径无效：{shard}")
    config = json.loads((model / "config.json").read_text())
    if config != identity["model_config"]:
        raise ValueError("checkpoint config 与历史权重不符")

    for name, expected in identity["tensors"].items():
        filename = expected["file"]
        if weights.get(name) != filename or Path(filename).name != filename:
            raise ValueError(f"checkpoint shard 映射不匹配：{name}")
        with (model / filename).open("rb") as stream:
            header_length = int.from_bytes(stream.read(8), "little")
            if not 0 < header_length <= 100_000_000:
                raise ValueError(f"safetensors 头部长度异常：{filename}")
            header = json.loads(stream.read(header_length))[name]
            first, last = header["data_offsets"]
            if (first < 0 or last - first != expected["bytes"]
                    or header["shape"] != expected["shape"] or header["dtype"] != expected["dtype"]):
                raise ValueError(f"权重 tensor 元数据不匹配：{name}")
            stream.seek(8 + header_length + first)
            remaining = last - first
            digest = hashlib.sha256()
            while remaining:
                chunk = stream.read(min(remaining, 8 << 20))
                if not chunk:
                    raise ValueError(f"权重文件被截断：{filename}")
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"权重 tensor SHA256 不匹配：{name}")
    return {
        "identity_sha256": sha(identity_path),
        "index_sha256": identity["index_sha256"],
        "tensor_bytes_sha_verified": True,
        "verified_tensors": sorted(identity["tensors"]),
        "model_dir": str(model.resolve()),
    }


def assemble(args):
    base, out = args.base_sglang.resolve(), args.output.resolve()
    if out.name != "runtime_sglang_g32" or out.exists() or base == out or base in out.parents:
        raise ValueError("输出必须是新建且与底座分离的 runtime_sglang_g32 目录")
    if args.source_checkout and args.model_dir:
        raise ValueError("公共源码 checkout 仅用于离线校验")
    check_source(base, "base_source_tree_sha256", args.source_checkout)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".g32-", dir=out.parent) as tmp:
        stage = Path(tmp) / out.name
        stage.mkdir()
        shutil.copytree(base, stage / "sglang", ignore=shutil.ignore_patterns("__pycache__"))
        for patch in sorted({entry["patch"] for entry in SPEC["changed_files"].values()}):
            with (ROOT / patch).open("rb") as stream:
                subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p1"],
                               cwd=stage, stdin=stream, check=True)
        check_source(stage / "sglang", "final_source_tree_sha256", args.source_checkout)
        for name, artifact in SPEC["runtime_additions"].items():
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / artifact["source"], dest)
            if sha(dest) != artifact["sha256"]:
                raise ValueError(f"G32 新增文件 SHA256 不符：{name}")
        if args.model_dir:
            proof = checkpoint_receipt(args.model_dir, stage / "nextn_checkpoint_identity.json")
            write_json(stage / "nextn_checkpoint_receipt.json", proof)
        shutil.copyfile(SPEC_PATH, stage / "g32_reproduction_manifest.json")
        os.rename(stage, out)
    print(f"G32 源码校验通过：{out}")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"G316 模板锚点已改变：{old}")
    return text.replace(old, new)


def configure(args):
    network = json.loads(args.network.read_text())
    p_ip, d_ip = (str(ipaddress.IPv4Address(network[role]["ip"])) for role in ("p", "d"))
    if p_ip == d_ip:
        raise ValueError("P/D 须使用不同的主机地址")
    role = args.role
    ip = p_ip if role == "p" else d_ip
    nic = network[role]["interface"]
    if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,15}", nic):
        raise ValueError("网卡名无效")
    task, model, coord = (path.resolve() for path in (args.task_dir, args.model_dir, args.coord_dir))
    runtime, run = (task / name for name in ("runtime_sglang_g32", "sg_pd_g316"))
    if not args.plan:
        proof = json.loads((runtime / "nextn_checkpoint_receipt.json").read_text())
        if proof["model_dir"] != str(model) or not proof["tensor_bytes_sha_verified"]:
            raise ValueError("先用同一权重目录重新组装 G32 并核对 NextN 权重")
        if sha(runtime / "g32_reproduction_manifest.json") != sha(SPEC_PATH):
            raise ValueError("G32 构建清单不匹配")
        check_source(runtime / "sglang", "final_source_tree_sha256")
        for name, artifact in SPEC["runtime_additions"].items():
            if sha(runtime / name) != artifact["sha256"]:
                raise ValueError(f"G32 运行文件已变化：{name}")
    cfg = json.loads((G316 / f"candidate_{role}.json").read_text())
    cfg["host_lan"] = ip
    cfg["argv"][cfg["argv"].index("--host") + 1] = "0.0.0.0"
    cfg["environment"].update(HCCL_IF_IP=ip, VLLM_HOST_IP=ip, HCCL_SOCKET_IFNAME=nic,
                              GLOO_SOCKET_IFNAME=nic, ASCEND_MF_STORE_URL=f"tcp://{p_ip}:8678")
    router = [
        "flock", "--nonblock", str(coord / f"{network['p']['hostname']}-port18080.lock"),
        "docker", "exec", "-w", "/task/sg_pd_g316", "yxy-glm53-sg-p-g316",
        "timeout", "--signal=TERM", "--kill-after=30", "43200",
        "python3", "-m", "sglang_router.launch_router", "--host", "0.0.0.0", "--port", "18080",
        "--pd-disaggregation", "--prefill", f"http://{p_ip}:19083", "8998",
        "--decode", f"http://{d_ip}:19084", "--max-concurrent-requests", "16",
        "--queue-size", "100", "--request-timeout-secs", "3600", "--disable-retries",
        "--health-check-endpoint", "/ready",
    ]
    if any(not re.fullmatch(r"[a-zA-Z0-9_.-]+", network[r]["hostname"]) for r in ("p", "d")):
        raise ValueError("主机名无效")
    controller = (G316 / "node_controller.py").read_text()
    for pattern, replacement in ((r"ROOT=[^;\n]+;", f"ROOT=Path({str(task)!r});"),
                                 (r"MODEL=Path\([^\n]*?\);", f"MODEL=Path({str(model)!r});")):
        controller, count = re.subn(pattern, lambda _: replacement, controller)
        if count != 1:
            raise ValueError(f"G316 模板变量锚点已改变：{pattern}")
    substitutions = (
        ("ROLE={'<ROLE_NODE>':'p','<ROLE_NODE>':'d'}[HOST]", f"ROLE={role!r}"),
        ("COORD=Path('<COORD_ROOT>')", f"COORD=Path({str(coord)!r})"),
        ("HOST not in ['<ROLE_NODE>','<ROLE_NODE>']", f"HOST != {network[role]['hostname']!r}"),
        ("sock.bind(('<HOST>',port))", "sock.bind(('0.0.0.0',port))"),
        ("'dev','enp67s0f0'", f"'dev',{nic!r}"),
        ("expected_ip='<HOST>' if ROLE=='p' else '<HOST>';assert expected_ip in ip.stdout",
         f"assert any(a['local']=={ip!r} for link in json.loads(ip.stdout) for a in link['addr_info'])"),
        ("runtime_sglang_g32/integrated_g18_manifest.json", "runtime_sglang_g32/g32_reproduction_manifest.json"),
    )
    for old, new in substitutions:
        controller = replace_once(controller, old, new)
    launcher = (G316 / "launch_pd.py").read_text()
    launcher = replace_once(launcher, "'tcp://<HOST>:8678'", repr(f"tcp://{p_ip}:8678"))
    old_manifest = next(line for line in launcher.splitlines() if line.startswith("m="))
    launcher = replace_once(launcher, old_manifest, """manifest = {
    'argv': argv,
    'candidate_sha256': hashlib.sha256((r/'candidate.json').read_bytes()).hexdigest(),
    'runtime_manifest_sha256': hashlib.sha256((runtime/'g32_reproduction_manifest.json').read_bytes()).hexdigest(),
    'environment': {k: os.environ[k] for k in c['environment']},
}
(r/'launch_manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')""")
    for name, content in (("node_controller.py", controller), ("launch_pd.py", launcher)):
        compile(content, name, "exec")
    if args.plan:
        print(json.dumps({"role": role, "run": str(run), "candidate": cfg, "router_argv": router}, indent=2))
        return

    if run.exists():
        raise ValueError(f"启动目录已经存在：{run}")
    with tempfile.TemporaryDirectory(prefix=".g316-", dir=task) as tmp:
        stage = Path(tmp) / run.name
        stage.mkdir()
        for name in ("native_readiness.py", "runtime_model_sha256.json", "run_pd.sh"):
            shutil.copyfile(G316 / name, stage / name)
        shutil.copytree(G316 / "glm53_native_gate", stage / "glm53_native_gate")
        (stage / "node_controller.py").write_text(controller)
        (stage / "launch_pd.py").write_text(launcher)
        write_json(stage / "candidate.json", cfg)
        if role == "p":
            write_json(stage / "router_argv.json", router)
            (stage / "run_router.sh").write_text("#!/bin/bash\nset -eu\nexec " + shlex.join(router) + "\n")
        os.rename(stage, run)
    print(f"G316 {role.upper()} 启动文件已生成：{run}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rebuild = sub.add_parser("assemble")
    for name in ("base-sglang", "output"):
        rebuild.add_argument("--" + name, required=True, type=Path)
    rebuild.add_argument("--model-dir", type=Path)
    rebuild.add_argument("--source-checkout", action="store_true")
    rebuild.set_defaults(handler=assemble)
    stage = sub.add_parser("configure")
    for name in ("task-dir", "model-dir", "coord-dir", "network"):
        stage.add_argument("--" + name, required=True, type=Path)
    stage.add_argument("--role", choices=("p", "d"), required=True)
    stage.add_argument("--plan", action="store_true")
    stage.set_defaults(handler=configure)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
