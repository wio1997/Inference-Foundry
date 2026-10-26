#!/usr/bin/env python3
"""Read-only current Draft Markov-bias task timeline; not a bound floor."""
import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = "12,256;129280,256"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    windows = []
    for rank in range(8):
        folder = Path(sorted(glob.glob(str(ROOT / f"evidence/20260926_loop060_resource/run246/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT")))[-1])
        events = json.loads((folder / "trace_view.json").read_text())
        proposer = sorted((x for x in events if x.get("cat") == "cpu_op" and x.get("name") == "extreme::proposer"), key=lambda x: x["ts"])
        target = sorted((x for x in events if x.get("cat") == "cpu_op" and x.get("name") == "extreme::target"), key=lambda x: x["ts"])
        assert len(proposer) == len(target) == 2
        kernels = list(csv.DictReader((folder / "kernel_details.csv").open(newline="")))
        for cycle, scope in enumerate(proposer):
            begin = float(scope["ts"])
            exit_host = begin + float(scope["dur"])
            tasks = sorted((x for x in kernels if x["Name"].startswith("aclnnMatmul_")
                            and x["Input Shapes"].strip(chr(34)) == INPUT
                            and begin < float(x["Start Time(us)"]) < exit_host + 6000),
                           key=lambda x: float(x["Start Time(us)"]))
            assert len(tasks) == 7
            starts = [float(x["Start Time(us)"]) for x in tasks]
            ends = [a + float(x["Duration(us)"]) for a, x in zip(starts, tasks)]
            assert all(starts[i + 1] >= ends[i] for i in range(6))
            read_gb = sum(float(x["aic_read_main_memory_datas(KB)"]) +
                          float(x["aiv_read_main_memory_datas(KB)"])
                          for x in tasks) * 1024 / 1e9
            windows.append({"rank": rank, "cycle": cycle,
                            "seven_task_read_counter_GB": read_gb,
                            "serial_task_span_us_profiled": ends[-1] - starts[0],
                            "task_duration_sum_us_profiled": sum(float(x["Duration(us)"]) for x in tasks),
                            "earliest_bias_start_us_vs_host_exit": starts[0] - exit_host,
                            "final_bias_end_us_vs_host_exit": ends[-1] - exit_host,
                            "next_target_host_start_us_vs_exit": float(target[cycle + 1]["ts"]) - exit_host if cycle + 1 < len(target) else None,
                            "inter_task_gaps_us": [starts[i + 1] - ends[i] for i in range(6)],
                            "stream_ids": sorted({x["Stream ID"] for x in tasks}),
                            "model_ids": sorted({x["Model ID"] for x in tasks})})
    spans = [x["serial_task_span_us_profiled"] for x in windows]
    durations = [x["task_duration_sum_us_profiled"] for x in windows]
    reads = [x["seven_task_read_counter_GB"] for x in windows]
    unique_bf16_weight_gb = 129280 * 256 * 2 / 1e9
    result = {"status": "profiled_current_markov_feedback_timeline_not_floor", "windows": windows,
              "summary": {"samples": len(windows), "span_us_range": [min(spans), max(spans)],
                          "span_us_median": statistics.median(spans),
                          "duration_sum_us_range": [min(durations), max(durations)],
                          "duration_sum_us_median": statistics.median(durations),
                          "seven_task_read_counter_GB_range": [min(reads), max(reads)],
                          "seven_task_read_counter_GB_median": statistics.median(reads),
                          "unique_BF16_bias_weight_GB": unique_bf16_weight_gb,
                          "read_counter_over_seven_weight_passes": statistics.median(reads) / (7 * unique_bf16_weight_gb)},
              "limits": ["Seven bias tasks have serial token feedback by source; inter-task gaps can contain required concurrent work, resource contention and profiler effects.",
                         "Host proposer exit is not Draft device completion; next Target Host entry is not first device consumer or a synchronization proof.",
                         "The 66.19MB unique Markov-bias matrix is used seven times; about0.464GB current task read is not automatically compulsory physical HBM, and caching/fusion or different algorithms change resource and schedule jointly.",
                         "Current profiled task duration and chain span are neither attainable latency floors nor removable Product wall-time."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
