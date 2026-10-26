# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Collect recently modified logs while preserving their directory layout."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class LogFile:
    path: Path
    relative_path: Path
    modified_time: datetime
    size_bytes: int


def _inspect_file(path: Path, source: Path) -> LogFile:
    stat_result = path.stat()
    return LogFile(
        path=path,
        relative_path=path.relative_to(source),
        modified_time=datetime.fromtimestamp(stat_result.st_mtime),
        size_bytes=stat_result.st_size,
    )


def scan_log_files(source: Path, label: str) -> list[LogFile]:
    records = []

    def report_error(error: OSError) -> None:
        print(f"WARNING: cannot scan {label}, path={error.filename}, error={error}", file=sys.stderr)

    for directory, dir_names, file_names in os.walk(source, followlinks=False, onerror=report_error):
        dir_names.sort()
        for name in sorted(file_names):
            path = Path(directory) / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                records.append(_inspect_file(path, source))
            except OSError as error:
                print(f"WARNING: cannot inspect {label}, path={path}, error={error}", file=sys.stderr)
    return records


def select_recent_files(files: list[LogFile], cutoff: datetime) -> list[LogFile]:
    selected = [record for record in files if record.modified_time >= cutoff]
    return sorted(selected, key=lambda record: record.relative_path)


def _format_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _write_summary(
    path: Path,
    label: str,
    source: Path,
    hours: int,
    cutoff: datetime,
    collected_at: datetime,
    scanned_count: int,
    selected_files: list[LogFile],
    size_bytes: int,
) -> None:
    lines = [
        f"{label} source: {source}",
        f"{label} window: last {hours} hours",
        f"{label} time range: {_format_time(cutoff)} - {_format_time(collected_at)}",
        f"{label} files scanned: {scanned_count}",
        f"{label} files selected: {len(selected_files)}",
        f"{label} selected size: {size_bytes} bytes",
    ]
    with path.open("a", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(lines) + "\n")


def _copy_files(files: list[LogFile], destination: Path, label: str) -> int:
    copied = 0
    for record in files:
        target = destination / record.relative_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.path, target)
            copied += 1
        except OSError as error:
            print(
                f"ERROR: failed to copy {label}, source={record.path}, destination={target}, error={error}",
                file=sys.stderr,
            )
    return copied


def _append_status(summary: Path, label: str, status: str) -> None:
    with summary.open("a", encoding="utf-8") as summary_file:
        summary_file.write(f"{label} status: {status}\n\n")


def collect_logs(source: Path, destination: Path, summary: Path, label: str, hours: int, max_mb: int) -> int:
    if not source.is_dir():
        raise NotADirectoryError(f"log source is not a directory: {source}")
    collected_at = datetime.now()
    cutoff = collected_at - timedelta(hours=hours)
    files = scan_log_files(source, label)
    selected_files = select_recent_files(files, cutoff)
    size_bytes = sum(record.size_bytes for record in selected_files)
    _write_summary(summary, label, source, hours, cutoff, collected_at, len(files), selected_files, size_bytes)
    if size_bytes > max_mb * 1024 * 1024:
        _append_status(summary, label, "skipped because selected files exceeded --max-size")
        print(f"WARNING: skip {label}: selected size exceeds limit, size={size_bytes}, max_mb={max_mb}")
        return 0
    copied = _copy_files(selected_files, destination, label)
    if copied != len(selected_files):
        _append_status(summary, label, f"partial copy ({copied}/{len(selected_files)} files)")
        print(f"WARNING: partial {label} copy, selected={len(selected_files)}, copied={copied}")
    else:
        _append_status(summary, label, "collected")
        print(f"[OK]    {label} ({size_bytes // 1024 // 1024} MB) -> {destination}")
    return 0


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--hours", type=_positive_integer, required=True)
    parser.add_argument("--max-size-mb", type=_positive_integer, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        return collect_logs(args.source, args.destination, args.summary, args.label, args.hours, args.max_size_mb)
    except (OSError, ValueError) as error:
        print(
            f"ERROR: failed to collect logs, label={args.label}, source={args.source}, "
            f"destination={args.destination}, error={error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
