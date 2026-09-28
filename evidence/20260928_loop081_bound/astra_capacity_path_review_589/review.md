# Independent Resource capacity-certificate path review during Run589

Read-only review, 2026-09-28. No service or device query/workload was invoked; no shared Run589 script, borrowed source, runtime state, frequency or module was changed. The current driver version file was read; firmware/board identity comes from admitted Run431/421, not a new live attestation. Current Formal remains 571.681 tok/s. Every strict finite endpoint remains unpromoted.

## Decision and newly useful evidence

The smallest useful next proof is **bind the installed A2 HCCS statistics API to its actual device-side implementation and counter time semantics before designing a short-window physical-traffic acquisition**. This is more informative now than another peak GEMM, d2d copy or HCCL throughput run. It can eliminate a invalid measurement method and enable a coarse-window link-traffic witness; it cannot by itself certify a Resource TPS limit.

The current installed source `dms_hccs_feature.c` defines a 500 ms refresh timer, keeps a per-device statistics cache, and copies that cache to the u64 query output. Its wrap extender reconstructs a u64 count from a low32 hardware register and detects one wrap by comparing the current low32 to its previous value. Six consecutive refresh failures stop the timer; until that threshold, a previous cache value may still be returned with the previous read_status. No hardware sample timestamp is present in the exported statistic structure. These are precise source-level facts, not observed timing of the actual device API.

**Applicability qualification:** `dms_hccs.mk` compiles this feature into the device-side ascend910B module. Host drv_devmng_host only includes the HCCS init/credit objects. The host-loaded drv_devmng_host srcversion therefore cannot certify this device implementation. The source-to-device image, installed-to-booted image, API dispatch and compiled branch remain unjoined. The installed device package contains `ascend_910b_device_sw.img` and `.bin`; reading their headers/manifests without loading them is a bounded next step. Do not declare 500 ms a measured or strict maximum staleness: the kernel timer can be delayed, and failed reads extend staleness.

The official A2 HDK26.1 npu-smi page states that HCCS reporting distinguishes standard link speed, current lane configuration and cumulative TX/RX statistics. It gives speed in Gb/s and a 20-byte count-unit relationship. This offers a traceable inventory/counter entry point. It does not attest maximum serializer clock including tolerance, bounded cache delay, payload semantics, full board topology, all alternative routes, or this installed26.0.rc1 build. The similarly named `dcmi_get_hccs_link_bandwidth_info` page found in search is for A3; a declaration in the shared header does not establish A2 support. Do not call it based only on that declaration.

## Exact environment and category separation

Admitted identity: Wuzhou S900K3; 8x910B3 IT21HMDC_Bin6, board0x62/PCB A/BOM1, 64 GiB/card, 20 Cube and40 Vector cores, architecture2201; driver26.0.rc1, firmware7.5.0.6.220, CANN9.1.0. The stored topology labels all off-diagonal pairs HCCS. This does not establish physical port-peer mapping or cut capacity. Configured Cube1800MHz and HBM1600MHz are not universal maxima.

| Evidence type | Meaning | Current example |
|---|---|---|
| Hard cumulative upper capacity | Service in every admitted interval is at most C+ times duration plus a justified burst/boundary allowance B | No matching exact-board certificate yet |
| Attained throughput/service | This fixture completed correctly at this rate | Run579/580 single and serial Graph arms |
| Concurrent contention | Co-running this admitted fixture changes service | Run579/580 concurrent whole chains lose; no universal overlap prohibition |
| Conditional Engineering projection | A declared schedule/resource model extrapolates measured feasible points | Must expose fixture transfer, resource coupling and confidence |

Run569 supplies conditional dense arithmetic, not fresh formal required counts. Run576 selected slices are stored-operand arithmetic, not compulsory memory reads. Run578 exports core-side main-memory counters, not physical HBM payload. Its failed paired timing cannot be filled with Run577 latency. Run579/580 are complete local all8 service witnesses, with private nonzero inputs and independent-ready branches, not a production dependency-legal schedule or universal capacity.

## Certificate paths by resource

**Compute:** under a named ordinary dense BF16 Cube-only class, the existing symbolic candidate is `8192 * sum_r(20*fmax_r)` conventional ops/s. It still needs an authoritative maximum clock/issue envelope for the exact bin/configuration, release/errata binding and interval convention. Other permitted engines require coverage or explicit class exclusion. W4A8 GEMM-equivalent work must not use the BF16 denominator. The required chip/OEM statement remains the Run462 exact-board request; the already tested rated-frequency and supported API shortcuts do not solve it. No finite microbenchmark sample can prove a universal maximum.

**HBM:** obtain authoritative active channel/bus width, transfers per clock, maximum allowed clock/tolerance, direction sharing, ECC/encoding treatment and matching physical transfer units. A safe raw-wire cap may deliberately omit protocol losses if raw capacity upper-bounds the counted payload; this relationship must be stated. A nominal memory clock, capacity in GiB, or core-side GM byte sum is insufficient. Compulsory bytes need a separately declared memory hierarchy/initial residency and legal recomputation/representation class. Weight footprint per cycle is not a proof. Installed AICore memory-rate config is a tuning model, not an exact-board controller certificate.

**HCCL/HCCS:** the primitive hardware certificate concerns directed physical edges/cuts, not a library's busbw metric. Need an all8 port-peer inventory, maximum lane/serializer envelope, clock tolerance and allowed alternate paths. For cut S, required cross-cut information divided by the sum of admitted directed edge upper rates is an optimistic communication relaxation. Do not add TX and RX of the same transfer or assume the current collective algorithm's bytes are mathematically required. TP8 fixes the parallel contract but does not automatically freeze materialization/collective boundaries. B must cover the chosen counting convention and boundary in-flight work. Standard link speed alone does not close those clauses.

## One bounded next proof / acquisition design

1. Offline, hash the installed CLI/DCMI/DSMI binaries and device images; inspect supported dispatch and package manifest/build identity. Join the HCCS statistics path to the archived device-side source or record the identity gap explicitly. No module load, firmware update, register write, debug probe, reset or service restart.
2. After the currently owned run completes, make one all8 supported `npu-smi info -t hccs -i R -c 0` inventory with command exit, host-monotonic start/end, version, lane mode/speed, all counters and status. This is a proposed future read-only acquisition, not performed here. Do not reset counters. If exact26.0 API support or output schema is absent, stop this path instead of trying undocumented subcommands.
3. Before any traffic attribution, establish whether output is hardware-immediate or cached, its count unit, width/wrap handling, reset behavior, update progress and error semantics. A stable idle value does not prove refresh. Source timer period is not a bounded refresh latency. If freshness cannot be independently witnessed, retain the inventory and mark per-cycle bytes unavailable.
4. Only if these semantics pass, a later already-planned long fixture may be enclosed by quiescent pre/post counter observations with all8 completion and observed post-completion refresh. Save host brackets and all rank completion; emit fixture-total observed link bytes, not a10ms-cycle rate. No arbitrary500ms sleep is a proof that the cache refreshed. If no timestamp/update-generation or equivalent supported freshness witness exists, require a device timestamped profiling collector instead.

This next proof narrows a real measurement-method uncertainty while leaving the exact C+/B and W-minus certificates separate. The new source warning prevents a short diagnostic from appearing to transfer zero bytes solely because both queries returned the same cache generation.

## Fail-closed gates and model effect

Reject strict capacity promotion for any of: HDK26.1 documentation used as installed26.0 behavior without a join; A3-only API used on A2; nominal/rated clock labelled maximum; missing positive clock tolerance; host module used to attest device implementation; unbounded timer/cache delay treated as B=0 or B=C*0.5s; stale counts; unaccounted counter resets/wraps; unknown port peer/direction; counted bytes outside the workload window; payload/20-byte-unit confusion; alternative permitted link/engine omitted; checkpoint footprint or current collective traffic labelled compulsory.

The right next model state is a source-backed HCCS measurement caveat plus explicit capacity-certificate requirements. It is not a new numerical ceiling. Keep strict endpoints null, preserve measured Engineering service columns and continue the fixed-W0 production dependency work. Do not repeat inaccessible OEM fetches or ordinary peak tests without a new primary source or a decision-critical attained-service uncertainty.

Primary new web source: https://www.hiascend.com/doc_center/source/zh/HDK/2610/A2/A2npu/npusmi_0075.html . A3 non-transfer example: https://www.hiascend.com/doc_center/source/zh/HDK/2610/A3/A3DCMI/dcmia3_149.html . Archived originals and every prior input SHA are in source_manifest.json.
