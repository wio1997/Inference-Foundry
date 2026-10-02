"""
Prefix cache hit rate collector.

Queries vLLM /metrics endpoint on each pod to collect prefix cache
queries/hits counters (HBM and external), computes hit rates between
two snapshots, and returns structured results suitable for CSV writing.
"""

import logging
import re
import subprocess
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _parse_pod_address(pod: str) -> tuple:
    """Parse a pod address string into (ip, port).

    Supports:
      - IPv4: "192.168.1.1:8000"
      - IPv6 with brackets: "[::1]:8000"
      - IPv6 without brackets: "fe80::1:8000"
    """
    if pod.startswith("["):
        # IPv6 with brackets: [ip]:port
        bracket_end = pod.index("]")
        ip = pod[1:bracket_end]
        port = pod[bracket_end + 2:]  # skip "]:"
        return ip, port
    elif pod.count(":") > 1:
        # IPv6 without brackets — last colon separates port
        ip, port = pod.rsplit(":", 1)
        return ip, port
    else:
        # IPv4: ip:port
        ip, port = pod.split(":")
        return ip, port


def _fetch_metrics(ip: str, port: str) -> str:
    """Fetch raw /metrics output from a vLLM service endpoint."""
    url = f"http://{ip}:{port}/metrics"
    command = (
        f"unset http_proxy && unset https_proxy && sleep 3s && curl -s {url}"
    )
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0 or not result.stdout.strip():
            logger.warning(f"No metrics data from {url}: {result.stderr}")
            return ""
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout fetching metrics from {url}")
        return ""
    except Exception as e:
        logger.warning(f"Error fetching metrics from {url}: {e}")
        return ""


def _parse_prefix_counters(metrics_text: str) -> Dict[int, Dict[str, int]]:
    """Parse prefix_cache_queries_total and prefix_cache_hits_total from metrics text.

    Returns:
        {engine_id: {"hbm_queries": X, "hbm_hits": X, "ext_queries": X, "ext_hits": X}}
    """
    result: Dict[int, Dict[str, int]] = {}

    for line in metrics_text.strip().split("\n"):
        # Only process lines with model_name label and prefix_cache metric
        if "prefix_cache" not in line or "model_name" not in line:
            continue

        # Extract engine id
        engine_match = re.search(r'engine="(\d+)"', line)
        if not engine_match:
            continue
        engine_id = int(engine_match.group(1))

        # Extract value (last numeric field)
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = float(parts[-1])
            if value.is_integer():
                value = int(value)
        except ValueError:
            continue

        # Initialize dict for this engine if not yet present
        if engine_id not in result:
            result[engine_id] = {
                "hbm_queries": 0, "hbm_hits": 0,
                "ext_queries": 0, "ext_hits": 0,
            }

        # Classify by metric name
        if "external_prefix_cache_queries_total" in line:
            result[engine_id]["ext_queries"] = value
        elif "external_prefix_cache_hits_total" in line:
            result[engine_id]["ext_hits"] = value
        elif "vllm:prefix_cache_queries_total" in line:
            result[engine_id]["hbm_queries"] = value
        elif "vllm:prefix_cache_hits_total" in line:
            result[engine_id]["hbm_hits"] = value

    return result


class HitRateCollector:
    """Collect and compute prefix cache hit rates from vLLM service endpoints.

    Usage:
        collector = HitRateCollector(pod_info=["10.0.0.1:8000", "10.0.0.2:8000"])
        before = collector.snapshot()
        # ... run test ...
        after = collector.snapshot()
        hit_rate = collector.compute_hit_rate(before, after)
    """

    def __init__(self, pod_info: list) -> None:
        """Args:
            pod_info: List of pod address strings, e.g. ["10.0.0.1:8000"].
        """
        self.pod_info = pod_info

    def snapshot(self) -> Dict[str, Dict[int, Dict[str, int]]]:
        """Collect current prefix cache counters from all pods.

        Returns:
            {pod_key: {engine_id: {"hbm_queries": X, "hbm_hits": X,
                                    "ext_queries": X, "ext_hits": X}}}
        """
        snapshot: Dict[str, Dict[int, Dict[str, int]]] = {}
        for pod in self.pod_info:
            ip, port = _parse_pod_address(pod)
            raw = _fetch_metrics(ip, port)
            counters = _parse_prefix_counters(raw)
            snapshot[pod] = counters
        return snapshot

    def compute_hit_rate(
        self,
        before: Dict[str, Dict[int, Dict[str, int]]],
        after: Dict[str, Dict[int, Dict[str, int]]],
    ) -> Dict:
        """Compute hit rates between two snapshots.

        Returns a structured dict with per-DP and aggregated rates:
        {
            "per_dp": {
                "dp0": {
                    "hbm_hit_rate": 0.85, "hbm_queries": 100, "hbm_hits": 85,
                    "ext_hit_rate": 0.10, "ext_queries": 50,  "ext_hits": 5,
                },
                "dp1": { ... },
            },
            "aggregated": {
                "hbm_hit_rate": 0.82, "hbm_queries": 200, "hbm_hits": 164,
                "ext_hit_rate": 0.08, "ext_queries": 100, "ext_hits": 8,
            },
        }

        Engine IDs from vLLM metrics are mapped to "dp0", "dp1", etc.
        Per-DP values are accumulated across pods (not overwritten).
        """
        per_dp: Dict[str, Dict] = {}
        agg_queries_hbm = 0
        agg_hits_hbm = 0
        agg_queries_ext = 0
        agg_hits_ext = 0

        for pod in self.pod_info:
            before_counters = before.get(pod, {})
            after_counters = after.get(pod, {})

            for engine_id in after_counters:
                b = before_counters.get(engine_id, {
                    "hbm_queries": 0, "hbm_hits": 0,
                    "ext_queries": 0, "ext_hits": 0,
                })
                a = after_counters[engine_id]

                queries_hbm = a["hbm_queries"] - b["hbm_queries"]
                hits_hbm = a["hbm_hits"] - b["hbm_hits"]
                queries_ext = a["ext_queries"] - b["ext_queries"]
                hits_ext = a["ext_hits"] - b["ext_hits"]

                dp_key = f"dp{engine_id}"

                # Accumulate per-DP values across pods (not overwrite)
                if dp_key not in per_dp:
                    per_dp[dp_key] = {
                        "hbm_queries": 0, "hbm_hits": 0,
                        "ext_queries": 0, "ext_hits": 0,
                    }
                per_dp[dp_key]["hbm_queries"] += queries_hbm
                per_dp[dp_key]["hbm_hits"] += hits_hbm
                per_dp[dp_key]["ext_queries"] += queries_ext
                per_dp[dp_key]["ext_hits"] += hits_ext

                agg_queries_hbm += queries_hbm
                agg_hits_hbm += hits_hbm
                agg_queries_ext += queries_ext
                agg_hits_ext += hits_ext

        # Compute per-DP hit rates from accumulated values
        for dp_key, dp_data in per_dp.items():
            hbm_rate = dp_data["hbm_hits"] / dp_data["hbm_queries"] if dp_data["hbm_queries"] > 0 else 0.0
            ext_rate = dp_data["ext_hits"] / dp_data["ext_queries"] if dp_data["ext_queries"] > 0 else 0.0
            per_dp[dp_key]["hbm_hit_rate"] = round(hbm_rate, 6)
            per_dp[dp_key]["ext_hit_rate"] = round(ext_rate, 6)

        agg_hbm_rate = agg_hits_hbm / agg_queries_hbm if agg_queries_hbm > 0 else 0.0
        agg_ext_rate = agg_hits_ext / agg_queries_ext if agg_queries_ext > 0 else 0.0

        aggregated = {
            "hbm_hit_rate": round(agg_hbm_rate, 6),
            "hbm_queries": agg_queries_hbm,
            "hbm_hits": agg_hits_hbm,
            "ext_hit_rate": round(agg_ext_rate, 6),
            "ext_queries": agg_queries_ext,
            "ext_hits": agg_hits_ext,
        }

        return {"per_dp": per_dp, "aggregated": aggregated}

    def print_hit_rate_table(self, hit_rate_info: Dict) -> None:
        """Print a formatted hit-rate table to console (for human readability)."""
        per_dp = hit_rate_info.get("per_dp", {})
        aggregated = hit_rate_info.get("aggregated", {})

        if not per_dp and not aggregated:
            logger.info("No hit rate data to display.")
            return

        col_w = 18
        total_w = col_w * 5 + 8

        print("\n" + "=" * total_w)
        print("Prefix Cache Hit Rate Summary")
        print("=" * total_w)

        headers = ["DP", "HBM Hit Rate", "HBM (hit/qry)", "Ext Hit Rate", "Ext (hit/qry)"]
        print(" ".join(f"{h:<{col_w}}" for h in headers))
        print("-" * total_w)

        for dp_key, dp_data in sorted(per_dp.items()):
            hbm_rate_str = f"{dp_data['hbm_hit_rate']:.2%}"
            hbm_detail = f"{dp_data['hbm_hits']}/{dp_data['hbm_queries']}"
            ext_rate_str = f"{dp_data['ext_hit_rate']:.2%}"
            ext_detail = f"{dp_data['ext_hits']}/{dp_data['ext_queries']}"
            print(f"{dp_key:<{col_w}} {hbm_rate_str:<{col_w}} {hbm_detail:<{col_w}} "
                  f"{ext_rate_str:<{col_w}} {ext_detail:<{col_w}}")

        print("-" * total_w)
        agg = aggregated
        hbm_rate_str = f"{agg['hbm_hit_rate']:.2%}"
        hbm_detail = f"{agg['hbm_hits']}/{agg['hbm_queries']}"
        ext_rate_str = f"{agg['ext_hit_rate']:.2%}"
        ext_detail = f"{agg['ext_hits']}/{agg['ext_queries']}"
        print(f"{'TOTAL':<{col_w}} {hbm_rate_str:<{col_w}} {hbm_detail:<{col_w}} "
              f"{ext_rate_str:<{col_w}} {ext_detail:<{col_w}}")
        print("=" * total_w)
