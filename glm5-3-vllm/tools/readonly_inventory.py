#!/usr/bin/env python3
"""readonly_inventory.py — 只读宿主机轻量采集（GLM-5.3 W8A8C8 / vLLM-Ascend 节点预检）

约束: 不写文件、不启停容器/服务、不初始化 NPU、不联网/SSH/git；不读 env/kubeconfig/密钥；
进程只取定向字段（无命令行）；docker 无 Env/Entrypoint/Cmd；无 kubectl（由离线侧另查）；
权重目录仅顶层 stat（不递归、不读内容）。用法（stdout 输出单个 JSON；脚本本身不写文件）:
    python3 readonly_inventory.py --label node32 > node32.json
    cat readonly_inventory.py | ssh node32 'python3 - --label node32' > node32.json
仅 Python 3.8+ 标准库。退出码: 0=已产出 JSON（个别探针失败只进 warnings）；2=致命错误。
"""
import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import threading
import time

try:
    import pwd
except ImportError:  # pragma: no cover - 非 POSIX
    pwd = None

SCHEMA = "readonly-inventory/2"
TARGET_IMAGE = "quay.io/ascend/vllm-ascend:v0.30.0rc1-a3-arm64"
WEIGHT_DIRS = ["/data/wio/GLM53/GLM-5.3-w8a8c8",
               "/data/wio/GLM53/.incoming-w8a8c8-w8a8c8-node32-33-20261017"]
CANN_VERSION_FILES = ["/usr/local/Ascend/driver/version.info", "/etc/ascend_install.info",
                      "/usr/local/Ascend/ascend-toolkit/latest/version.info",
                      "/usr/local/Ascend/ascend-toolkit/latest/ascend_toolkit_install.info",
                      "/usr/local/Ascend/firmware/version.info",
                      "/usr/local/Ascend/nnal/atb/latest/version.info"]
DEFAULT_TIMEOUT, DEFAULT_MAX_BYTES = 20.0, 262144

# 定向字段模板: docker 只取身份/状态列，绝不 dump Config.Env 或容器 Command。
PS_FIELDS = "pid,ppid,user,comm,pcpu,pmem,etime"
SYSTEMCTL_PROPS = ("LoadState,ActiveState,SubState,UnitFileState,MainPID,ExecMainPID,"
                   "ExecMainStartTimestamp,NRestarts,Result")
DOCKER_PS_FORMAT = "table {{.ID}}|{{.Image}}|{{.Names}}|{{.Status}}|{{.Ports}}|{{.RunningFor}}"
DOCKER_IMAGE_FORMAT = ('{"Id":{{json .Id}},"RepoTags":{{json .RepoTags}},"RepoDigests":{{json .RepoDigests}},'
                       '"Created":{{json .Created}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"Size":{{json .Size}}}')
DOCKER_INFO_FORMAT = ("ServerVersion={{.ServerVersion}} Architecture={{.Architecture}} KernelVersion={{.KernelVersion}} "
                      "NCPU={{.NCPU}} MemTotal={{.MemTotal}} Driver={{.Driver}} DefaultRuntime={{.DefaultRuntime}} "
                      "DockerRootDir={{.DockerRootDir}} Name={{.Name}}")
SH_PCI_NUMA = ("for d in /sys/bus/pci/devices/*; do "
               "echo \"${d##*/} numa_node=$(cat \"$d/numa_node\" 2>/dev/null)\"; done")
SH_NET = ("for n in /sys/class/net/*; do echo \"${n##*/} operstate=$(cat \"$n/operstate\" 2>/dev/null) speed_mbps=$(cat \"$n/speed\" 2>/dev/null) "
          "numa_node=$(cat \"$n/device/numa_node\" 2>/dev/null) device=$(readlink -f \"$n/device\" 2>/dev/null)\"; done")
SH_CANN = ("for f in " + " ".join(CANN_VERSION_FILES) + "; do if [ -f \"$f\" ]; then "
           "echo \"== $f\"; head -c 2048 \"$f\"; echo; fi; done")
SH_WEIGHTS = ("for d in " + " ".join(WEIGHT_DIRS) + "; do echo \"== $d\"; "
              "ls -lnA --time-style=full-iso \"$d\" 2>&1 | head -n 300; done")

PROBES = [
    ("hostname", ["hostname"], "节点主机名"),
    ("uname", ["uname", "-a"], "内核/架构/版本"),
    ("date_utc", ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], "宿主机 UTC 时间"),
    ("lscpu", ["lscpu"], "CPU/拓扑/NUMA/cache"),
    ("numactl_hardware", ["numactl", "--hardware"], "NUMA 节点/距离矩阵/各节点内存"),
    ("free", ["free", "-h"], "内存/swap 概览"),
    ("df", ["df", "-hT"], "挂载点用量/文件系统类型"),
    ("lspci_tree", ["lspci", "-tv"], "PCI 拓扑树"),
    ("ip_route", ["ip", "route"], "IPv4 路由表"),
    ("npu_smi_usage", ["npu-smi"], "npu-smi 官方 usage/帮助（不推测子命令）"),
    ("npu_smi_info", ["npu-smi", "info"], "NPU 设备/进程/占用总览"),
    ("docker_ps", ["docker", "ps", "--no-trunc", "--format", DOCKER_PS_FORMAT], "运行中容器（定向列，无 Command）"),
    ("docker_image_identity", ["docker", "image", "inspect", "--format", DOCKER_IMAGE_FORMAT, TARGET_IMAGE],
     "目标镜像 ID/架构等身份字段（无 Env/Entrypoint/Cmd）"),
    ("docker_info", ["docker", "info", "--format", DOCKER_INFO_FORMAT], "docker 守护进程定向字段"),
    ("ps_fields", ["ps", "-eo", PS_FIELDS], "进程定向字段（无命令行）"),
    ("systemctl_show_kubelet", ["systemctl", "show", "kubelet", "-p", SYSTEMCTL_PROPS], "kubelet 单元状态"),
    ("systemctl_show_docker", ["systemctl", "show", "docker", "-p", SYSTEMCTL_PROPS], "docker 单元状态"),
    ("sysfs_pci_numa", ["sh", "-c", SH_PCI_NUMA], "PCI 设备 NUMA 亲和（-1=无亲和）"),
    ("sysfs_net_numa_speed", ["sh", "-c", SH_NET], "网卡速率/NUMA/PCI 归属（sysfs）"),
    ("cann_version_files", ["sh", "-c", SH_CANN], "驱动/CANN 版本标识（白名单文件，非密钥）"),
    ("weight_dirs_top_level", ["sh", "-c", SH_WEIGHTS], "两个权重目录顶层 stat（不递归、单目录 300 行上限）"),
]


def utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def one_line(text, limit=160):
    lines = (text or "").strip().splitlines()
    line = lines[0] if lines else ""
    return line if len(line) <= limit else line[:limit] + "..."


def _drain(pipe, cap, sink):
    chunks, total = [], 0
    try:
        while True:
            chunk = pipe.read(65536)
            if not chunk:
                break
            total += len(chunk)
            room = cap - (total - len(chunk))
            if room > 0:
                chunks.append(chunk[:room])
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass
    sink["truncated"] = total > cap
    sink["text"] = b"".join(chunks).decode("utf-8", "replace")


def _kill(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=3.0)
            return
        except Exception:
            pass


def run_command(name, argv, purpose, timeout, max_bytes):
    """执行一条只读探针；失败/超时/截断都写进记录，不抛出。"""
    rec = {"name": name, "purpose": purpose, "argv": list(argv), "timeout_s": timeout,
           "max_bytes_per_stream": max_bytes, "start_utc": utc_now_iso(), "end_utc": None,
           "duration_s": None, "returncode": None, "timed_out": False, "stdout": "", "stderr": "",
           "stdout_truncated": False, "stderr_truncated": False, "spawn_error": None}
    t0 = time.monotonic()
    try:
        # start_new_session: 超时可整组终止; stdin=DEVNULL: 脚本可能来自 stdin，子进程不得读它。
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)
    except Exception as exc:
        rec["spawn_error"] = "%s: %s" % (type(exc).__name__, exc)
        rec["end_utc"], rec["duration_s"] = utc_now_iso(), round(time.monotonic() - t0, 3)
        return rec
    sinks = ({}, {})
    threads = [threading.Thread(target=_drain, args=(proc.stdout, max_bytes, sinks[0])),
               threading.Thread(target=_drain, args=(proc.stderr, max_bytes, sinks[1]))]
    for thread in threads:
        thread.daemon = True
        thread.start()
    try:
        rec["returncode"] = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        rec["timed_out"] = True
        _kill(proc)
        rec["returncode"] = proc.returncode
    for thread in threads:
        thread.join(5.0)
    rec["stdout"], rec["stderr"] = sinks[0].get("text", ""), sinks[1].get("text", "")
    rec["stdout_truncated"] = bool(sinks[0].get("truncated"))
    rec["stderr_truncated"] = bool(sinks[1].get("truncated"))
    rec["end_utc"], rec["duration_s"] = utc_now_iso(), round(time.monotonic() - t0, 3)
    return rec


def build_parser():
    parser = argparse.ArgumentParser(
        prog="readonly_inventory.py",
        description=("只读、有界的宿主机轻量采集：按 PROBES 逐条执行定向只读命令，stdout 输出单个 JSON。"
                     "不写文件、不启停服务/容器、不初始化 NPU、不联网；不读 env/kubeconfig/密钥；"
                     "不输出进程命令行。"),
        epilog="每条命令独立超时与字节上限，失败记录后继续；最坏耗时约 timeout x 探针数。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="", metavar="TEXT", help="写入 JSON 的标签（如 node32）")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, metavar="SECONDS",
                        help="每条命令的墙钟超时秒数（默认 %(default)s）")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, metavar="N",
                        help="每条命令 stdout/stderr 各自的采集上限字节（默认 %(default)s）")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.timeout <= 0 or args.max_bytes < 1024:
        sys.stderr.write("--timeout 需大于 0；--max-bytes 至少 1024\n")
        return 2
    wall0 = time.monotonic()
    commands, warnings = [], []
    for name, cmd_argv, purpose in PROBES:
        rec = run_command(name, cmd_argv, purpose, args.timeout, args.max_bytes)
        commands.append(rec)
        if rec["spawn_error"]:
            warnings.append("%s: 命令不可用（%s）" % (name, rec["spawn_error"]))
        elif rec["timed_out"]:
            warnings.append("%s: 超时 %.1fs 后被终止，输出可能不完整" % (name, args.timeout))
        elif rec["returncode"] not in (0, None):
            warnings.append("%s: 退出码 %s（%s）" % (name, rec["returncode"], one_line(rec["stderr"])))
        if rec["stdout_truncated"] or rec["stderr_truncated"]:
            warnings.append("%s: 输出被 %d 字节上限截断" % (name, args.max_bytes))
    uid = os.getuid() if hasattr(os, "getuid") else None
    if pwd is None or uid is None:
        user = None
    else:
        try:
            user = pwd.getpwuid(uid).pw_name
        except Exception:
            user = None
    inventory = {
        "schema": SCHEMA,
        "label": (args.label or "")[:200],
        "generated_utc": utc_now_iso(),
        "duration_s": round(time.monotonic() - wall0, 3),
        "collector": {"script": "glm5-3-vllm/tools/readonly_inventory.py", "argv": sys.argv,
                      "python": sys.version.split()[0], "uid": uid, "user": user,
                      "timeout_s_per_command": args.timeout, "max_bytes_per_stream": args.max_bytes},
        "commands": commands,
        "warnings": warnings,
    }
    try:
        sys.stdout.buffer.write((json.dumps(inventory, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.stderr.write("中断\n")
        sys.exit(130)
    except Exception as exc:  # 兜底: stdout 上仍给出可解析的 JSON
        sys.stdout.write(json.dumps({"schema": SCHEMA, "argv": sys.argv, "generated_utc": utc_now_iso(),
                                     "fatal_error": "%s: %s" % (type(exc).__name__, exc)},
                                    ensure_ascii=False) + "\n")
        sys.exit(2)
