# SPDX-License-Identifier: Apache-2.0
"""Exact R247 B service command, shared by the host probe and CPU checks.

The command is a pure function of node, evidence root, overlay manifest and
the frozen R201 argv.  It never starts anything by itself.  The host probe
embeds this source verbatim before creating the container.
"""
import pathlib
import shlex

RUN_ID = "r247"
NAMESPACE = "glm53-flash-r247"
IMAGE = "sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263"
MODEL_PATH = "/data/wio/models/GLM-5.3-Flash-w8a8"
OWNER = "zcode"
CONTAINER_PREFIX = "if-glm53-flash-r247-b"
NODES = (46, 47)
ROLES = {46: "Prefill", 47: "Decode"}
IPS = {46: "172.16.12.46", 47: "172.16.12.47"}
PORTS = {46: [(0, 19030), (46, 19031), (46, 19032), (46, 19024)],
         47: [(0, 19030), (47, 19031)] + [(47, 19100 + i) for i in range(16)]}


def container_name(node):
    if node not in NODES:
        raise ValueError("R247 is restricted to nodes 46/47")
    return CONTAINER_PREFIX + str(node)


def service_environment(node):
    if node not in NODES:
        raise ValueError("R247 is restricted to nodes 46/47")
    ip = IPS[node]
    env = {
        "FLASH_B103_PERSISTENT_OWNED_STORAGE": "1" if node == 47 else "0",
        "FLASH_B103_OWNED_COMPRESSION": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "RAYON_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "1",
        "SGLANG_OPT_USE_TILELANG_MHC_PRE": "0",
        "SGLANG_OPT_USE_TILELANG_MHC_POST": "0",
        "HCCL_DETERMINISTIC": "true",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "400",
        "HCCL_IF_IP": ip,
        "SGLANG_HOST_IP": ip,
        "FLASH_PD_ROLE": ROLES[node],
        "FLASH_SDMA_VERIFY_BYTES": "0",
        "ASCEND_MF_STORE_URL": "tcp://172.16.12.46:19024",
        "ASCEND_MF_TRANSFER_PROTOCOL": "sdma",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "10800",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "10800",
        "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE": "1",
        "SGLANG_DISAGGREGATION_QUEUE_SIZE": "1",
        "SGLANG_RUST_SERVER": "0",
        "FLASH_PD_EXACT_COMPUTE_BUDGET": "1",
        "FLASH_PD_PREFILL_COHORT_GATE": "0",
        "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE": "1",
        "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT": "2",
        "FLASH_PD_COMPRESSION_ABI_AUDIT": "0",
        "FLASH_PD_SCORE_ABI_AUDIT": "0",
        "FLASH_PD_GRACEFUL_RECV": "1",
        "FLASH_MHC_FIXED_ROWS": "0",
        "FLASH_PD_STATE_AUDIT": "0",
        "FLASH_PD_PERF_MODE": "1",
        "FLASH_PD_STARTUP_DTYPE_META": "1",
        "FLASH_PD_LAYOUT_ONCE": "0",
        "FLASH_KDA_PREFIX_ONCE_AUDIT": "0",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    if node == 46:
        env.update({
            "R234_PROFILE_OBSERVER": "1",
            "R234_PROFILE_ROOT": "/run-evidence/profile-B",
            "R234_PROFILE_FRAMEWORK": "sglang",
            "R234_PROFILE_NODE": "46",
            "R234_PROFILE_ROLE": "P",
            "R234_PROFILE_WINDOWS": "early:8192:16384,mid:32768:40960,late:61440:73728",
            "R234_PROFILE_MAX_FORWARDS": "6",
            "R234_PROFILE_MAX_METADATA_FORWARDS": "64",
            "R234_PROFILE_MAX_WINDOW_BYTES": "4294967296",
            "R234_PROFILE_MIN_FREE_BYTES": "8589934592",
            "R234_PROFILE_EXPECT_ACTIVE": "1",
            "R234_PROFILE_ARM_FILE": "/run-evidence/profile-B/ARMED",
            "R234_OBSERVER_ROOT": "/run-evidence/observer",
            "R234_OBSERVER_MANIFEST": "/run-evidence/observer/manifest-sglang.json",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
    return env


def build(node, evidence_root, overlay_rows, argv, verify_host_paths=True,
          model_path=MODEL_PATH):
    """Return the exact ``docker run`` argv for one owned B node."""
    if node not in NODES:
        raise ValueError("R247 is restricted to nodes 46/47")
    root = pathlib.Path(evidence_root)
    model = pathlib.Path(model_path)
    name = container_name(node)
    args = [
        "docker", "run", "-d",
        "--log-opt", "max-size=256m", "--log-opt", "max-file=2",
        "--pull=never", "--name", name,
        "--label", "project_id=glm53-flash",
        "--label", "run_id=" + RUN_ID,
        "--label", "namespace=" + NAMESPACE,
        "--label", "owner=" + OWNER,
        "--network=host", "--ipc=private", "--shm-size=32g",
        "--ulimit", "memlock=-1:-1", "--memory=512g", "--pids-limit=4096",
        "-v", str(root) + ":/run-evidence",
        "-v", str(model) + ":" + str(model) + ":ro",
    ]
    for row in overlay_rows:
        target = row["target"]
        if pathlib.PurePosixPath(target).is_absolute() or ".." in pathlib.PurePosixPath(target).parts:
            raise ValueError("Unsafe overlay target")
        args += ["-v", str(root / "overlay" / target)
                 + ":/sgl-workspace/sglang/python/" + target + ":ro"]
    for key, value in service_environment(node).items():
        args += ["-e", key + "=" + value]
    if verify_host_paths:
        for path in ("/usr/local/Ascend/driver", "/usr/local/dcmi",
                     "/etc/ascend_install.info", "/usr/local/bin/npu-smi"):
            if pathlib.Path(path).exists():
                args += ["-v", path + ":" + path + ":ro"]
    for index in range(16):
        path = "/dev/davinci" + str(index)
        if verify_host_paths:
            if not pathlib.Path(path).exists():
                raise FileNotFoundError(path)
        args += ["--device", path]
    if verify_host_paths:
        for path in ("/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc"):
            if not pathlib.Path(path).exists():
                raise FileNotFoundError(path)
            args += ["--device", path]
    serve = ["python3", "/run-evidence/start_after_preflight.py"] + list(argv)
    launch = "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    if node == 46:
        launch += (" && export PYTHONPATH=/run-evidence/observer/sglang:"
                   "/run-evidence/observer/common${PYTHONPATH:+:$PYTHONPATH}")
    launch += " && exec " + shlex.join(serve)
    args += ["--entrypoint", "/bin/bash", IMAGE, "-c", launch]
    return args
