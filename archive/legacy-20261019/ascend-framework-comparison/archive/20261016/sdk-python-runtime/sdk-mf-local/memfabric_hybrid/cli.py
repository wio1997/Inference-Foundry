# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Command-line for memfabric-hybrid"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from . import mem_scan
from ._acc_offload_build import ExtendBuildError as _ExtendBuildError
from ._acc_offload_build import acc_offload_info as _acc_offload_info
from ._acc_offload_build import ensure_acc_offload_ext as _ensure_acc_offload_ext
from ._acc_offload_build import uninstall_acc_offload_ext as _uninstall_acc_offload_ext
from .env_utils import get_soc_key as _get_soc_key

LOGGER = logging.getLogger(__name__)


def _show_version(_args: argparse.Namespace) -> int:
    version_file = Path(__file__).resolve().parent / "VERSION"
    try:
        sys.stdout.write(version_file.read_text(encoding="utf-8"))
    except OSError as error:
        LOGGER.error("Failed to read version file, path=%s, error=%s", version_file, error)
        return 1
    return 0


def _install_acc_offload(soc_key=None) -> int:
    """Build and install the acc_offload extend library after kernel ops install.

    Any failure fails the whole kernel install. Only the "environment does not
    support acc_offload (SoC/CANN/torch)" skip counts as success. All progress
    and skip messages are echoed by _acc_offload_build itself.
    """
    try:
        _ensure_acc_offload_ext(soc_key)
    except _ExtendBuildError as exc:
        LOGGER.error("failed to install acc_offload operator extend library: %s", exc)
        return 1
    except OSError as exc:
        LOGGER.error("failed to install acc_offload operator extend library, unexpected OS error: %s", exc)
        return 1
    return 0


def _manage_kernel(args: argparse.Namespace) -> int:
    if args.command == "version":
        return _show_version(args)

    bash = shutil.which("bash")
    if bash is None:
        LOGGER.error("bash is required to install kernel ops.")
        return 1

    installer = Path(__file__).resolve().parent / "_kernel" / "hybm" / "install.sh"
    if not installer.is_file():
        LOGGER.error(
            "kernel ops installer is missing, path=%s. Build the wheel in a valid CANN environment.", installer
        )
        return 1

    soc_key = None
    if args.command == "install":
        soc_key = args.soc_version or _get_soc_key()
        if soc_key == "A3":
            LOGGER.info("SoC A3 has no AICPU kernel support, skip kernel ops install.")
            return _install_acc_offload(soc_key)

    command = [bash, str(installer), f"--{args.command}"]
    if args.command == "install" and args.install_for_all:
        command.append("--install-for-all")
    exit_code = subprocess.run(command, check=False).returncode
    if exit_code != 0:
        return exit_code
    if args.command == "install":
        return _install_acc_offload(soc_key)
    if args.command == "uninstall":
        _uninstall_acc_offload_ext()
    if args.command == "info":
        _acc_offload_info()
    return 0


def _scan_memory(args: argparse.Namespace) -> int:
    mem_scan.show(args.node, args.min_mb, args.workers)
    return 0


def _register_version_command(components) -> None:
    version = components.add_parser("version", help="show version information")
    version.set_defaults(handler=_show_version)


def _register_kernel_command(components) -> None:
    kernel = components.add_parser("kernel", help="manage kernel")
    kernel.set_defaults(handler=_manage_kernel)
    commands = kernel.add_subparsers(dest="command", required=True)

    install = commands.add_parser("install", help="install kernel ops into CANN")
    install.add_argument(
        "--install-for-all",
        action="store_true",
        help="make the installed files readable by all users",
    )
    install.add_argument(
        "--soc-version",
        choices=("A2", "A3", "A5"),
        default=None,
        help="target SoC generation; skip auto detection when given (default: auto detect)",
    )
    commands.add_parser("info", help="show kernel ops install paths and CANN load configuration")
    commands.add_parser("version", help="show installed kernel version")
    commands.add_parser("uninstall", help="uninstall kernel")


def _register_mem_command(components) -> None:
    mem = components.add_parser("mem", help="manage memory")
    commands = mem.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="scan NUMA free memory")
    mem_scan.add_arguments(scan)
    scan.set_defaults(handler=_scan_memory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mfcli")
    components = parser.add_subparsers(dest="component", required=True)
    _register_version_command(components)
    _register_kernel_command(components)
    _register_mem_command(components)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    args = _parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
