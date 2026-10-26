"""Summarize independently accepted certificates; local files only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = {}
    rows = []
    pending = []
    for concurrency in (16, 32, 64):
        for framework in ("vllm", "sglang"):
            relative = (
                f"results/vllm-formal-C{concurrency}-v2-measurement-v1.json"
                if framework == "vllm" else
                f"results/sglang-L04-C{concurrency}-v1-measurement-v1.json"
            )
            path = ROOT / relative
            if not path.is_file():
                pending.append({"framework": framework, "concurrency": concurrency})
                continue
            raw = path.read_bytes()
            certificate = json.loads(raw)
            assert certificate["measurement_valid"] is True
            assert certificate["project_id"] == "glm53"
            assert certificate["framework"] == framework
            assert certificate["concurrency"] == concurrency
            assert certificate["formal_requests"] == 4 * concurrency
            assert certificate["formal_successful_output_tokens"] == 2400 * concurrency
            assert certificate["client_audit_pass"] is True
            assert certificate["native_postrun_audit_required"] is False
            native_path = certificate["native_postrun_evidence"]
            native_raw = (ROOT / native_path).read_bytes()
            assert hashlib.sha256(native_raw).hexdigest() == certificate["native_postrun_evidence_sha256"]
            sources[relative] = hashlib.sha256(raw).hexdigest()
            sources[native_path] = hashlib.sha256(native_raw).hexdigest()
            quantiles = certificate["quantiles"]
            rows.append({
                "framework": framework,
                "concurrency": concurrency,
                "algorithm": "L04 metadata batch" if framework == "sglang" else "unchanged vLLM",
                "run_id": certificate["run_id"],
                "formal_requests": certificate["formal_requests"],
                "formal_successful_output_tokens": certificate["formal_successful_output_tokens"],
                "full_clock_seconds": certificate["full_clock_seconds"],
                "output_tokens_per_second": certificate["output_tokens_per_second"],
                "TTFT_seconds": quantiles["ttft_seconds"],
                "TPOT_ms": {key: value * 1000 for key, value in quantiles["tpot_seconds"].items()},
                "aggregate_SLA_pass": certificate["aggregate_SLA_pass"],
                "certificate": relative,
            })
    paired = []
    for concurrency in (16, 32, 64):
        pair = {row["framework"]: row for row in rows if row["concurrency"] == concurrency}
        if len(pair) == 2:
            paired.append({
                "concurrency": concurrency,
                "vllm_over_optimized_sglang_throughput_ratio": pair["vllm"]["output_tokens_per_second"] / pair["sglang"]["output_tokens_per_second"],
                "scope": "End-to-end deployment comparison; not a pure framework attribution or a before/after optimization estimate.",
            })
    result = {
        "project_id": "glm53",
        "complete_matched_C16_C32_C64": not pending,
        "contract": "Ordinary W8A8 GLM-5.3, separate 2P2D deployments, SDMA, 80000 input/600 output, four fresh suffixes per independent prefix, uniform fresh-HTTP v2 client, full first-dispatch to last-terminal clock.",
        "rows": rows,
        "matched_deployment_ratios": paired,
        "pending": pending,
        "optimization_before_after": "results/sglang-L04-C16-observed-before-after-v1.json",
        "limitations": [
            "One accepted run per listed condition; repeated full-run noise has not been estimated.",
            "C16 has a separate unoptimized SGLang baseline. C32/C64 do not establish a SGLang optimization delta.",
            "Native AISBench steady-window metrics remain separate from the full-clock primary throughput.",
            "Valid measurement and fixed latency SLA acceptance are separate.",
            "Hardware groups, native graph/cache/SDK packaging and staging differ between deployments; no pure framework or machine-independent attribution.",
            "Historical invalid vLLM C64 v1 remains excluded from accepted rows.",
        ],
        "source_evidence_sha256": sources,
    }
    destination = ROOT / "results/L04-matched-deployment-comparison-v1.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(destination), "accepted_rows": len(rows), "complete": not pending, "pending": pending}))


if __name__ == "__main__":
    main()
