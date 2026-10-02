"""
Result writer for prefix cache performance tests.

Parses AISBench log output for performance metrics, collects hit-rate data,
and writes results to both CSV and JSONL formats. Hit-rate columns are
included per-DP and aggregated, for both HBM and external caches.
"""

import csv
import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AISBench log parser
# ---------------------------------------------------------------------------

def parse_aisbench_log(log_path: str, request_rate: str = "0", npu_num: int = 1) -> Dict:
    """Parse an AISBench performance log file and extract metrics.

    Returns a dict with standardized key names. Missing metrics default to -1.
    """
    defaults = {
        "total_requests": -1,
        "max_concurrency": -1,
        "measured_concurrency": -1,
        "ttft_avg_ms": -1,
        "ttft_p90_ms": -1,
        "tpot_avg_ms": -1,
        "tpot_p90_ms": -1,
        "benchmark_duration_s": -1,
        "output_token_throughput": -1,
        "single_output_throughput": -1,
        "input_token_throughput": -1,
        "total_token_throughput": -1,
        "single_total_throughput": -1,
        "prefill_token_throughput": -1,
        "request_throughput_qps": -1,
        "request_throughput_qpm": -1,
        "total_input_tokens": -1,
        "total_output_tokens": -1,
    }
    result = defaults.copy()
    result["request_rate"] = request_rate
    log_dir = ""

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            # Extract log directory
            m = re.search(r"Current exp folder:\s*(.+)$", line)
            if m:
                log_dir = m.group(1).strip()

            # TTFT
            if "TTFT" in line and "Time To First Token" not in line:
                nums = re.findall(r"(\d+\.\d+)", line)
                if nums:
                    vals = list(map(float, nums))
                    result["ttft_avg_ms"] = vals[0] if len(vals) > 0 else -1
                    result["ttft_p90_ms"] = vals[5] if len(vals) > 5 else -1

            # TPOT
            if "TPOT" in line and "Time Per Output Token" not in line:
                nums = re.findall(r"(\d+\.\d+)", line)
                if nums:
                    vals = list(map(float, nums))
                    result["tpot_avg_ms"] = vals[0] if len(vals) > 0 else -1
                    result["tpot_p90_ms"] = vals[5] if len(vals) > 5 else -1

            # Benchmark Duration
            if "Benchmark Duration" in line:
                nums = re.findall(r"(\d+\.\d+)", line)
                if nums:
                    result["benchmark_duration_s"] = float(nums[0]) / 1000

            # Concurrency (measured)
            if "Concurrency" in line and "Max Concurrency" not in line:
                nums = re.findall(r"(\d+\.\d+)", line)
                if nums:
                    result["measured_concurrency"] = float(nums[0])

            # Max Concurrency
            if "Max Concurrency" in line:
                parts = re.findall(r"[\w']+", line)
                if parts:
                    result["max_concurrency"] = parts[-1]

            # Output Token Throughput
            if "Output Token Throughput" in line:
                nums = re.findall(r"(\d+\.\d+)", line)
                if nums:
                    result["output_token_throughput"] = float(nums[0])
                    result["single_output_throughput"] = float(nums[0]) / npu_num

            # Input Token Throughput
            if "Input Token Throughput" in line:
                nums = re.findall(r"(\d+\.\d+)", line)
                if nums:
                    result["input_token_throughput"] = float(nums[0])

            # Total Token Throughput
            if "Total Token Throughput" in line:
                nums = re.findall(r"(\d+\.\d+)", line)
                if nums:
                    result["total_token_throughput"] = float(nums[0])
                    result["single_total_throughput"] = float(nums[0]) / npu_num

            # InputTokens
            if "InputTokens" in line:
                nums = re.findall(r"(\d+\.?\d*)", line)
                if nums:
                    result["total_input_tokens"] = float(nums[0])

            # OutputTokens
            if "OutputTokens" in line:
                nums = re.findall(r"(\d+\.?\d*)", line)
                if nums:
                    result["total_output_tokens"] = float(nums[0])

            # Total Requests
            if "Total Requests" in line and "Request Throughput" not in line:
                nums = re.findall(r"(\d+\.?\d*)", line)
                if nums:
                    result["total_requests"] = float(nums[0])

            # Request Throughput
            if "Request Throughput" in line:
                nums = re.findall(r"(\d+\.\d+)", line)
                if nums:
                    result["request_throughput_qps"] = float(nums[0])
                    result["request_throughput_qpm"] = float(nums[0]) * 60

            # Prefill Token Throughput
            if "Prefill Token Throughput" in line:
                nums = re.findall(r"(\d+\.\d+)", line)
                if nums:
                    result["prefill_token_throughput"] = float(nums[0])

    except FileNotFoundError:
        logger.error(f"AISBench log file not found: {log_path}")
    except Exception as e:
        logger.error(f"Error parsing AISBench log: {e}")

    return result, log_dir


# ---------------------------------------------------------------------------
# Archive log
# ---------------------------------------------------------------------------

def archive_log(source_log: str, log_dir: str) -> None:
    """Copy the AISBench log to its experiment directory and append to all-logs file."""
    if not os.path.exists(source_log):
        logger.warning(f"Source log {source_log} not found, skipping archive.")
        return

    # Copy to experiment directory
    if log_dir:
        try:
            os.system(f"cp {source_log} {log_dir}")
        except Exception as e:
            logger.warning(f"Failed to copy log to {log_dir}: {e}")

    # Append to cumulative log
    all_log_path = "aisbench_all.log"
    try:
        with open(source_log, "r", encoding="utf-8") as src:
            content = src.read()
        with open(all_log_path, "a", encoding="utf-8") as tgt:
            tgt.write(f"\n\n{'=' * 50}\n")
            tgt.write(f"{'=' * 50}\n\n")
            tgt.write(content)
            tgt.write(f"\n\n{'=' * 50}\n")
            tgt.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            tgt.write(f"{'=' * 50}\n")
        logger.info(f"Log archived to {all_log_path}")
    except Exception as e:
        logger.error(f"Failed to archive log: {e}")


# ---------------------------------------------------------------------------
# Build result row dict
# ---------------------------------------------------------------------------

def build_result_row(
    perf_metrics: Dict,
    hit_rate_info: Dict,
    args,
    phase: str = "full",
    round_index: int = 1,
) -> Dict:
    """Build a single result row dict combining performance metrics, hit rates, and config.

    Args:
        perf_metrics: Dict from parse_aisbench_log().
        hit_rate_info: Dict from HitRateCollector.compute_hit_rate().
        args: argparse namespace with test configuration.
        phase: "warmup" or "full" to indicate which test phase.
        round_index: Round number (1-based) for multi-round tests.

    Returns:
        A flat dict suitable for CSV/JSONL writing.
    """
    row = {
        # Timestamp and config
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "round": round_index,
        "test_name": getattr(args, "test_name", "") or "",
        "phase": phase,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "data_num": args.data_num,
        "concurrency": args.concurrency,
        "request_rate": args.request_rate,
        "dp": args.dp,
        "prefix_num": args.prefix_num,
        "repeat_rate": args.repeat_rate,
        "seed": args.seed,
        "model_path": args.model_path,
        "host_ip": args.host_ip,
        "host_port": args.host_port,
        "url": args.url,
        "npu_num": args.npu_num,

        # Performance metrics
        "total_requests": perf_metrics.get("total_requests", -1),
        "max_concurrency": perf_metrics.get("max_concurrency", -1),
        "measured_concurrency": perf_metrics.get("measured_concurrency", -1),
        "ttft_avg_ms": perf_metrics.get("ttft_avg_ms", -1),
        "ttft_p90_ms": perf_metrics.get("ttft_p90_ms", -1),
        "tpot_avg_ms": perf_metrics.get("tpot_avg_ms", -1),
        "tpot_p90_ms": perf_metrics.get("tpot_p90_ms", -1),
        "benchmark_duration_s": perf_metrics.get("benchmark_duration_s", -1),
        "output_token_throughput": perf_metrics.get("output_token_throughput", -1),
        "single_output_throughput": perf_metrics.get("single_output_throughput", -1),
        "input_token_throughput": perf_metrics.get("input_token_throughput", -1),
        "total_token_throughput": perf_metrics.get("total_token_throughput", -1),
        "single_total_throughput": perf_metrics.get("single_total_throughput", -1),
        "prefill_token_throughput": perf_metrics.get("prefill_token_throughput", -1),
        "request_throughput_qps": perf_metrics.get("request_throughput_qps", -1),
        "request_throughput_qpm": perf_metrics.get("request_throughput_qpm", -1),
        "total_input_tokens": perf_metrics.get("total_input_tokens", -1),
        "total_output_tokens": perf_metrics.get("total_output_tokens", -1),
    }

    # ---- Add per-DP hit rate columns ----
    per_dp = hit_rate_info.get("per_dp", {})
    for dp_key, dp_data in sorted(per_dp.items()):
        row[f"hbm_hit_rate_{dp_key}"] = dp_data.get("hbm_hit_rate", 0.0)
        row[f"hbm_queries_{dp_key}"] = dp_data.get("hbm_queries", 0)
        row[f"hbm_hits_{dp_key}"] = dp_data.get("hbm_hits", 0)
        row[f"ext_hit_rate_{dp_key}"] = dp_data.get("ext_hit_rate", 0.0)
        row[f"ext_queries_{dp_key}"] = dp_data.get("ext_queries", 0)
        row[f"ext_hits_{dp_key}"] = dp_data.get("ext_hits", 0)

    # ---- Add aggregated hit rate columns ----
    agg = hit_rate_info.get("aggregated", {})
    row["hbm_hit_rate_total"] = agg.get("hbm_hit_rate", 0.0)
    row["hbm_queries_total"] = agg.get("hbm_queries", 0)
    row["hbm_hits_total"] = agg.get("hbm_hits", 0)
    row["ext_hit_rate_total"] = agg.get("ext_hit_rate", 0.0)
    row["ext_queries_total"] = agg.get("ext_queries", 0)
    row["ext_hits_total"] = agg.get("ext_hits", 0)

    return row


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(row: Dict, csv_path: str) -> None:
    """Write or append a result row to a CSV file.

    If the file exists, the row is appended (columns are aligned by name).
    If not, a new file is created with all columns from *row*.

    When appending, new columns that didn't exist before are added with
    empty values for prior rows, ensuring forward compatibility with
    varying DP counts.
    """
    file_exists = os.path.exists(csv_path)

    if file_exists:
        # Read existing headers
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_headers = next(reader)

        # Merge headers: keep existing order, add new columns at end
        new_cols = [k for k in row.keys() if k not in existing_headers]
        all_headers = existing_headers + new_cols

        # Read existing data rows
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

        # Re-write with merged headers + existing rows + new row
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_headers, extrasaction="ignore")
            writer.writeheader()
            for old_row in existing_rows:
                writer.writerow(old_row)
            writer.writerow(row)

        logger.info(f"Appended result row to {csv_path} (new cols: {new_cols})")
    else:
        # Create new file
        headers = list(row.keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerow(row)
        logger.info(f"Created new result CSV: {csv_path}")


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def write_jsonl(row: Dict, jsonl_path: str) -> None:
    """Append a result row to a JSONL file (one JSON object per line)."""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, default=str)
        f.write("\n")
    logger.info(f"Appended result row to {jsonl_path}")


# ---------------------------------------------------------------------------
# JSONL → CSV converter (for CI / batch processing)
# ---------------------------------------------------------------------------

def jsonl_to_csv(jsonl_path: str, csv_path: Optional[str] = None) -> str:
    """Convert a JSONL results file to CSV.

    Args:
        jsonl_path: Path to the JSONL file.
        csv_path: Optional output CSV path. Defaults to same name with .csv extension.

    Returns:
        Path to the generated CSV file.
    """
    if csv_path is None:
        csv_path = jsonl_path.rsplit(".", 1)[0] + ".csv"

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON: {e}")

    if not records:
        logger.warning(f"No valid records in {jsonl_path}")
        return csv_path

    # Collect all field names preserving insertion order
    fieldnames = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Converted {len(records)} records from {jsonl_path} → {csv_path}")
    return csv_path
