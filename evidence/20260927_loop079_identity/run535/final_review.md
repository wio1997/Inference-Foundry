# Run535 final review — PASS for guarded Host-lineage acquisition

**PASS**, for the exact final source snapshot recorded in `final_inputs_sha256.json`. This supersedes the live-readiness FAIL in `astra_live_readiness.md` and `addendum.md`; their findings remain historical evidence. The approved action is one guarded instrumented warmup48 + measured48 acquisition. This is preflight admission, not a successful live run or a correctness/performance verdict. No live source mutation, service start or NPU execution was performed by this reviewer.

## Final reviewed bytes

- Controller: `38141d6918c6f8437c8de8efc7b95d85d6498958d4d3ddfa723b35d6f81ab793`.
- Patch generator: `3dd36594271405bfb852327ea0e0200d9ddabda0eab4454d7241794fab31f64f`.
- Host helper: `5c381122b9b1781bad2733c2f4b1e296ff82e929baf2de247225085983f3df20`.
- Phase barrier: `738e60788373afec85eca992e7ddb8cd6545f9021dc11d21a53bcd2a02ac6424`.
- Server reducer: `88e5997e1e4edcab840e1b792eb8aa6c3dbf5907d62f9b81f0b363c5de595dee`.
- Final admission: `ea093143aa603e79543d23da57cb9f21db9acfac1f34294756147d249eefe7f7`.

## Corrected admission chain

The controller requires an exclusive lock and fresh run directory, verifies idle process and eight-card HBM state before installation, and propagates failed probes explicitly even during cleanup. INT/TERM route through cleanup. Source installation/restoration uses durable backups, same-directory temporary writes, fsync and atomic replacement. Restore verifies all five original pins and exact manifest key set, accepts only known original/patched hashes, and remains available after helper drift. Helper drift invalidates acquisition even if restoration succeeds.

The warmup barrier joins exactly48 unique client external IDs to internal IDs and the same ordered12 requests on all8×4 handoff/drain/done records. It validates Runtime output length, Host mirror/FULL mode, terminal Scheduler status/G arithmetic, Output identities, actual API external fields, contiguous per-process sequences and paired flush digests/identities. The measured phase marker is changed only after this barrier exits0.

The final reducer reconstructs ordinary Scheduler admissions followed by the clipped Runtime terminal prefix, including nonzero prebulk G, stable generation identity, full incoming Runtime equality and terminal stopped state. It joins that complete1024-token sequence through Output receive/queue and API raw consumption/cumulative counts; it independently compares every framed server SSE payload to the received client payload. The warmup-only client request index must exactly equal the warmup half of the final two-phase client report. Both phases require exact request coverage, all8 cohort membership and Runtime parity, complete flush records and phase identities. Measured Runner FULL is checked explicitly. Controller imports use the real repository/CANN environment and were exercised without launching vLLM.

Final admission runs after service stop, idle verification, restoration and before/after source/script SHA comparisons. It rejects nonzero mandatory stages, helper drift, changed manifests or inputs. Its completed status file can be independently replayed after the terminal admission/final exit fields are appended.

## Independent verification

- Corrected patch CPU fixture: exit0; interrupted temporary write, mixed partial installation, helper-drift recovery, exact-source-key negatives and truncation checks.
- Final server reducer selftest: exit0, **1 positive +10 negative** cases; positive includes nonzero G with distinct prebulk token and Runtime clipping, on64 rank/cohort records across eight rank PIDs.
- Final admission selftest: exit0, **2 positive +8 negative** cases, including replay after terminal status fields.
- Independent `run535/barrier_probe.py`: exit0, **1 positive +5 negative** cases covering wrong API ID, non-FULL Runner, stale Scheduler, wrong Output identity and wrong footer run.
- Controller `bash -n`: exit0.
- Container CANN/ATB environment import with `sys.path[0]` changed to the CLI bin directory: exit0; helper resolved from `/data/wio/Inference_Foundry/scripts/loop079_formal_ledger_server.py` through retained repository PYTHONPATH.

## Accepted scope

Controller cross-process clock identity remains unobserved, and the reducer intentionally removed comparisons using the controller's monotonic timestamp; that limitation is explicit in `unresolved`. Phase admission relies on the sequential successful barrier/transition controller and per-event marker identity. Host cycle coverage, monotonic progress and park transitions are checked, but exact Host-progress/count-history alignment remains unresolved and must not be promoted into a scheduling-time bound.

Per-event file reads, JSON allocation and fsync perturb execution. A passing live acquisition can establish its Host raw-token/phase/API lineage; it cannot by itself establish fresh compulsory Target multiplicity, device-ready prefill/seed/KV, mixed-resource attainable service, typed collective completion or a Product lower-bound DAG. Formal Current remains571.681tok/s and this preflight changes no finite Resource, Hardware, Scheduling or Product endpoint. If any live gate fails, retain the acquisition as failed diagnostic evidence, stop/restore, and inspect the existing data before deciding whether another run is necessary.
