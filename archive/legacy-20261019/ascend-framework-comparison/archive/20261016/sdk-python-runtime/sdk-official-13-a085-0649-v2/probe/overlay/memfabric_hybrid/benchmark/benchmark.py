# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Benchmark entry point for mfcli benchmark command."""

import logging
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

LOGGER = logging.getLogger(__name__)


def _is_localhost(store_url):
    host = store_url.split("://")[-1].split(":")[0]
    return host in ("127.0.0.1", "localhost")


def _count_from_npu_smi(output):
    """Count NPUs from `npu-smi info -l` output."""
    total = 0
    ids = 0
    for line in (output or "").splitlines():
        text = line.strip()
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        key = key.strip().upper()
        if key.startswith("TOTAL COUNT"):
            found = re.search(r"\d+", value)
            if found:
                total = int(found.group(0))
        elif key == "NPU ID":
            ids += 1
    return total or ids


def _parse_npu_devices():
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if visible:
        devices = [int(x.strip()) for x in visible.split(",") if x.strip()]
        if devices:
            return devices

    npu_smi = shutil.which("npu-smi")
    if npu_smi:
        try:
            proc = subprocess.run([npu_smi, "info", "-l"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            proc = None
        if proc is not None:
            count = _count_from_npu_smi(proc.stdout)
            if count > 0:
                return list(range(count))

    # /dev/davinci0..N-1. The numeric suffix filter matters: a plain "davinci*"
    # glob also matches /dev/davinci_manager and would report one NPU too many.
    davinci_devs = [dev for dev in Path("/dev").glob("davinci*") if dev.name[len("davinci") :].isdigit()]
    if davinci_devs:
        return list(range(len(davinci_devs)))

    return [0]


def _configure_benchmark(args):
    if args.world_size == -1:
        args.world_size = args.local_rank_size if _is_localhost(args.ip) else 2 * args.local_rank_size
    args.paired = args.local_rank_size > 0 and args.world_size == 2 * args.local_rank_size
    if args.paired:
        _configure_paired(args)


def _configure_paired(args):
    if args.command not in ("write", "read"):
        raise ValueError("paired mode requires write or read")
    endpoint = urlsplit(args.ip)
    if (
        endpoint.scheme != "tcp"
        or not endpoint.hostname
        or endpoint.port is None
        or endpoint.username is not None
        or endpoint.path
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("paired --ip must be tcp://<write-host>:<port>")
    if endpoint.hostname == "0.0.0.0" or not 0 < endpoint.port < 65536:
        raise ValueError("paired --ip must have a reachable host and a valid port")
    if args.command == "read" and args.rank_start == 0:
        args.rank_start = args.local_rank_size
    expected_start = 0 if args.command == "write" else args.local_rank_size
    if args.rank_start != expected_start:
        raise ValueError(f"paired {args.command} requires --rank-start {expected_start}")
    if args.sync_port is None:
        args.sync_port = endpoint.port + 1
    if not 0 < args.sync_port < 65536 or args.sync_port == endpoint.port:
        raise ValueError("--sync-port must be valid and different from the BM store port")
    if not math.isfinite(args.sync_timeout) or args.sync_timeout <= 0 or args.warmup_count < 0:
        raise ValueError("--sync-timeout must be positive and finite; --warmup-count must be nonnegative")
    if args.copysize % 4096:
        raise ValueError("paired --copysize must be a multiple of 4096 bytes")
    if args.peer_mem_type == "device" and args.protocol.startswith("host_"):
        raise ValueError("host protocols in paired mode require --peer-mem-type host")
    capacity = args.local_hbm_size if args.peer_mem_type == "device" else 4 * 1024**3
    if args.local_hbm_size < 0 or args.copysize * args.batchsize > capacity:
        raise ValueError("copy payload exceeds the contributed pool size, or HBM size is negative")
    args.sync_host = endpoint.hostname


def _validate_layout(args, npu_count):
    if args.local_rank_size <= 0 or args.device < 0 or args.device + args.local_rank_size > npu_count:
        raise ValueError("requested local ranks/devices exceed the visible NPU count")
    if args.copysize <= 0 or args.copysize % 4 or args.batchsize <= 0 or args.copy_count <= 0:
        raise ValueError("copy size must be positive and divisible by 4; batch size and copy count must be positive")
    if args.rank_start < 0 or args.rank_start + args.local_rank_size > args.world_size:
        raise ValueError("local rank range must be within --world-size")


def _log_invalid_configuration(args, error):
    LOGGER.error(
        "invalid benchmark configuration: %s; rank_start=%s local_ranks=%s world_size=%s "
        "device=%s protocol=%s copy_size=%s batch_size=%s copy_count=%s",
        error,
        args.rank_start,
        args.local_rank_size,
        args.world_size,
        args.device,
        args.protocol,
        args.copysize,
        args.batchsize,
        args.copy_count,
    )


def run(args):
    devices = _parse_npu_devices()
    npu_count = len(devices)

    if args.device is None:
        args.device = 0
    if args.local_rank_size is None:
        args.local_rank_size = npu_count

    try:
        _configure_benchmark(args)
        _validate_layout(args, npu_count)
    except ValueError as error:
        _log_invalid_configuration(args, error)
        return 1

    LOGGER.info(
        "benchmark: cmd=%s mode=%s protocol=%s framework=%s model=%s npu_devices=%s world_size=%d local_rank_size=%d",
        args.command,
        "paired" if args.paired else "same_operation",
        args.protocol,
        args.framework,
        args.model,
        devices,
        args.world_size,
        args.local_rank_size,
    )

    if args.command in ("write", "read"):
        from .bm_band_width_benchmark import run_bm

        return run_bm(args)
    LOGGER.error("unknown benchmark command: %s", args.command)
    return 1
