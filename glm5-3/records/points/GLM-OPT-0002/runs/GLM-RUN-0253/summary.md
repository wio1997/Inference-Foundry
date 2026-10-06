# Run253 — same-stream MoE event code comparison

Actual controller completed15:36:29Z; eventcompare phase exit0/no timeout. Planned manifest and original running Zcode handoff remain immutable. Only owned D251 retired; P249 unchanged; native16×11 cases passed before one D253 load. No new profile, layout/config scan, H4 stacking, kernel change or large workload.

Same-resident A1/B1/A2/B2: one excluded warm, two timed2334prompt/8token golden requests and one complete natural-EOS request per mode. Native176 byte/dependency/import/lifetime cases,64 resident worker mode witnesses, exact golden IDs/chunks/full KV hits and same complete sentence/23IDs/chunks/EOS all passed. Pure candidate helper AST matches executed diagnostic shim; other four shim sources are pure candidate.

| Pair | Short median TPOT ms | Complete TPOT ms | Complete D wall s | Complete PD wall s |
|---|---:|---:|---:|---:|
|A1/B1|250.891/243.907|229.346/218.544|5.451206/5.187993|5.922158/5.587866|
|A2/B2|264.363/244.450|226.595/219.210|5.382251/5.198689|5.771890/5.586090|

Complete generation TPOT reductions4.71/3.26%; D-wall reductions4.83/3.41%. P contributes71.079/2.239ms to total PD reductions and must not be credited to D code. Baseline complete D drift68.956ms versus paired D savings263.214/183.561ms; baseline TPOT drift2.751ms versus10.802/7.385ms paired change. Repeated positive scoped code evidence. Four complete requests and MTP two-token chunks are not SLA percentiles/capacity/full-workload acceptance. Formal PERF_KEEP remains INCONCLUSIVE/PARKED; Current=None, stack empty. [Sol decision](../../research/decode_path_20261006/H5_DECISION.md).

Finally selector0 and original five disk sources restored; D retains stock behavior in the resident diagnostic shim. Fresh read-only15:48:54Z health/idle/32owner/source hashes and exited controller confirmed. [Offline original SSE/timing validator](../../research/decode_path_20261006/reduce_event_compare.py), [execution summary](execution_summary.json), [compact original-arrival timeline](matched_event_timeline.json). Raw client events, native log and complete source artifacts remain on servers, indexed by path/size/hash.
